package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
)

type node struct {
	Leaf        *float64 `json:"leaf,omitempty"`
	Feature     string   `json:"feature,omitempty"`
	Threshold   float64  `json:"threshold,omitempty"`
	DefaultLeft bool     `json:"default_left,omitempty"`
	Left        *node    `json:"left,omitempty"`
	Right       *node    `json:"right,omitempty"`
}

type model struct {
	BaseScore       float64             `json:"base_score"`
	FeatureDefaults map[string]float64  `json:"feature_defaults"`
	Trees           []*node             `json:"trees"`
	Categories      map[string][]string `json:"categories"`
}

func loadModel(path string) (*model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m model
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

type manifest struct {
	Version  int64  `json:"version"`
	SHA256   string `json:"sha256"`
	ModelURL string `json:"modelUrl"`
}

func fetchManifest(client *http.Client, url string) (*manifest, error) {
	resp, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("manifest fetch: unexpected status %d", resp.StatusCode)
	}
	var m manifest
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, err
	}
	return &m, nil
}

func fetchModelVerified(client *http.Client, url, expectedSHA256 string) (*model, error) {
	resp, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("model fetch: unexpected status %d", resp.StatusCode)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256(data)
	got := hex.EncodeToString(sum[:])
	if got != expectedSHA256 {
		return nil, fmt.Errorf("sha256 mismatch: manifest says %s, downloaded bytes hash to %s", expectedSHA256, got)
	}
	var m model
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

func evalTree(n *node, features map[string]float64) float64 {
	if n.Leaf != nil {
		return *n.Leaf
	}
	value, ok := features[n.Feature]
	goLeft := n.DefaultLeft
	if ok {
		goLeft = float32(value) < float32(n.Threshold)
	}
	if goLeft {
		return evalTree(n.Left, features)
	}
	return evalTree(n.Right, features)
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (m *model) predict(features map[string]float64) float64 {
	full := make(map[string]float64, len(m.FeatureDefaults)+len(features))
	for k, v := range m.FeatureDefaults {
		full[k] = v
	}
	for k, v := range features {
		full[k] = v
	}
	margin := m.BaseScore
	for _, t := range m.Trees {
		margin += evalTree(t, full)
	}
	return sigmoid(margin)
}

func (m *model) encodeRequest(payload map[string]any) map[string]float64 {
	features := make(map[string]float64, len(payload))
	for key, raw := range payload {
		if categories, isCategorical := m.Categories[key]; isCategorical {
			strValue, _ := raw.(string)
			features[key] = float64(indexOf(categories, strValue))
			continue
		}
		if num, ok := raw.(float64); ok {
			features[key] = num
		}
	}
	return features
}

func indexOf(values []string, target string) int {
	for i, v := range values {
		if v == target {
			return i
		}
	}
	return -1
}

func bucketVerdict(attackProbability float64) string {
	switch {
	case attackProbability < 0.4:
		return "allow"
	case attackProbability > 0.6:
		return "block"
	default:
		return "uncertain"
	}
}

type loadedInfo struct {
	Version int64
	SHA256  string
}

type server struct {
	currentModel atomic.Pointer[model]
	currentInfo  atomic.Pointer[loadedInfo]
	apiKey       string
	manifestURL  string
	httpClient   *http.Client
}

func (s *server) reloadIfChanged() (bool, *loadedInfo, error) {
	if s.manifestURL == "" {
		return false, nil, fmt.Errorf("no manifest URL configured")
	}
	m, err := fetchManifest(s.httpClient, s.manifestURL)
	if err != nil {
		return false, nil, err
	}
	if current := s.currentInfo.Load(); current != nil && current.SHA256 == m.SHA256 {
		return false, current, nil
	}
	newModel, err := fetchModelVerified(s.httpClient, m.ModelURL, m.SHA256)
	if err != nil {
		return false, nil, err
	}
	info := &loadedInfo{Version: m.Version, SHA256: m.SHA256}
	s.currentModel.Store(newModel)
	s.currentInfo.Store(info)
	return true, info, nil
}

func (s *server) requireAuth(c *gin.Context) bool {
	if s.apiKey == "" {
		return true
	}
	if c.GetHeader("Authorization") != "Bearer "+s.apiKey {
		c.JSON(http.StatusUnauthorized, gin.H{"detail": "invalid or missing API key"})
		c.Abort()
		return false
	}
	return true
}

func (s *server) health(c *gin.Context) {
	info := s.currentInfo.Load()
	body := gin.H{
		"status":       "ok",
		"model_loaded": s.currentModel.Load() != nil,
	}
	if info != nil {
		body["version"] = info.Version
		body["sha256"] = info.SHA256
	}
	c.JSON(http.StatusOK, body)
}

func (s *server) reload(c *gin.Context) {
	if !s.requireAuth(c) {
		return
	}
	changed, info, err := s.reloadIfChanged()
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"detail": err.Error()})
		return
	}
	body := gin.H{"changed": changed}
	if info != nil {
		body["version"] = info.Version
		body["sha256"] = info.SHA256
	}
	c.JSON(http.StatusOK, body)
}

func (s *server) analyzeNetwork(c *gin.Context) {
	if !s.requireAuth(c) {
		return
	}
	m := s.currentModel.Load()
	if m == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"detail": "model not loaded"})
		return
	}

	payload := map[string]any{}
	if c.Request.ContentLength != 0 {
		if err := json.NewDecoder(c.Request.Body).Decode(&payload); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"detail": "invalid JSON body"})
			return
		}
	}

	attackProbability := m.predict(m.encodeRequest(payload))
	c.JSON(http.StatusOK, gin.H{
		"verdict":      bucketVerdict(attackProbability),
		"safety_score": math.Round((1.0-attackProbability)*10000) / 10000,
		"reasoning":    nil,
	})
}

func main() {
	s := &server{
		apiKey:      os.Getenv("NOXOS_INFERENCE_API_KEY"),
		manifestURL: os.Getenv("NOXOS_MANIFEST_URL"),
		httpClient:  &http.Client{Timeout: 30 * time.Second},
	}

	if s.manifestURL != "" {
		if _, info, err := s.reloadIfChanged(); err != nil {
			log.Printf("initial model load from manifest %s failed: %v (starting anyway, /health will report it)", s.manifestURL, err)
		} else {
			log.Printf("loaded model version=%d sha256=%s from manifest", info.Version, info.SHA256)
		}
	} else {
		modelPath := envOr("NOXOS_MODEL_PATH", "model.json")
		m, err := loadModel(modelPath)
		if err != nil {
			log.Printf("model not loaded from %s: %v (starting anyway, /health will report it)", modelPath, err)
		} else {
			data, _ := os.ReadFile(modelPath)
			sum := sha256.Sum256(data)
			s.currentModel.Store(m)
			s.currentInfo.Store(&loadedInfo{Version: 0, SHA256: hex.EncodeToString(sum[:])})
			log.Printf("loaded model from local file %s", modelPath)
		}
	}

	if s.manifestURL != "" {
		if intervalSeconds, err := strconv.Atoi(envOr("NOXOS_RELOAD_INTERVAL_SECONDS", "300")); err == nil && intervalSeconds > 0 {
			go func() {
				ticker := time.NewTicker(time.Duration(intervalSeconds) * time.Second)
				defer ticker.Stop()
				for range ticker.C {
					changed, info, err := s.reloadIfChanged()
					if err != nil {
						log.Printf("periodic reload check failed: %v", err)
						continue
					}
					if changed {
						log.Printf("reloaded model version=%d sha256=%s", info.Version, info.SHA256)
					}
				}
			}()
		}
	}

	router := gin.Default()
	router.GET("/health", s.health)
	router.POST("/reload", s.reload)
	router.POST("/analyze/network", s.analyzeNetwork)

	addr := ":" + envOr("PORT", "8080")
	log.Printf("listening on %s", addr)
	log.Fatal(router.Run(addr))
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
