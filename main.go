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
	Cover       float64  `json:"cover"`
	Left        *node    `json:"left,omitempty"`
	Right       *node    `json:"right,omitempty"`
}

type model struct {
	BaseScore       float64             `json:"base_score"`
	FeatureDefaults map[string]float64  `json:"feature_defaults"`
	FeatureScale    map[string]float64  `json:"feature_scale"`
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

func (m *model) mergedFeatures(features map[string]float64) map[string]float64 {
	full := make(map[string]float64, len(m.FeatureDefaults)+len(features))
	for k, v := range m.FeatureDefaults {
		full[k] = v
	}
	for k, v := range features {
		full[k] = v
	}
	return full
}

func (m *model) predict(features map[string]float64) float64 {
	full := m.mergedFeatures(features)
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

type slot struct {
	name        string
	manifestURL string
	localPath   string
	model       atomic.Pointer[model]
	info        atomic.Pointer[loadedInfo]
}

func (sl *slot) configured() bool {
	return sl.manifestURL != "" || sl.localPath != ""
}

func (sl *slot) reloadIfChanged(client *http.Client) (bool, *loadedInfo, error) {
	if sl.manifestURL == "" {
		return false, nil, fmt.Errorf("no manifest URL configured for the %s model", sl.name)
	}
	m, err := fetchManifest(client, sl.manifestURL)
	if err != nil {
		return false, nil, err
	}
	if current := sl.info.Load(); current != nil && current.SHA256 == m.SHA256 {
		return false, current, nil
	}
	newModel, err := fetchModelVerified(client, m.ModelURL, m.SHA256)
	if err != nil {
		return false, nil, err
	}
	info := &loadedInfo{Version: m.Version, SHA256: m.SHA256}
	sl.model.Store(newModel)
	sl.info.Store(info)
	return true, info, nil
}

func (sl *slot) loadInitial(client *http.Client) {
	switch {
	case sl.manifestURL != "":
		if _, info, err := sl.reloadIfChanged(client); err != nil {
			log.Printf("initial %s model load from manifest %s failed: %v (starting anyway, /health will report it)", sl.name, sl.manifestURL, err)
		} else {
			log.Printf("loaded %s model version=%d sha256=%s from manifest", sl.name, info.Version, info.SHA256)
		}
	case sl.localPath != "":
		m, err := loadModel(sl.localPath)
		if err != nil {
			log.Printf("%s model not loaded from %s: %v (starting anyway, /health will report it)", sl.name, sl.localPath, err)
			return
		}
		data, _ := os.ReadFile(sl.localPath)
		sum := sha256.Sum256(data)
		sl.model.Store(m)
		sl.info.Store(&loadedInfo{Version: 0, SHA256: hex.EncodeToString(sum[:])})
		log.Printf("loaded %s model from local file %s", sl.name, sl.localPath)
	}
}

func (sl *slot) startPeriodicReload(client *http.Client, interval time.Duration) {
	if sl.manifestURL == "" {
		return
	}
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for range ticker.C {
			changed, info, err := sl.reloadIfChanged(client)
			if err != nil {
				log.Printf("periodic %s reload check failed: %v", sl.name, err)
				continue
			}
			if changed {
				log.Printf("reloaded %s model version=%d sha256=%s", sl.name, info.Version, info.SHA256)
			}
		}
	}()
}

func (sl *slot) status() gin.H {
	body := gin.H{"loaded": sl.model.Load() != nil}
	if info := sl.info.Load(); info != nil {
		body["version"] = info.Version
		body["sha256"] = info.SHA256
	}
	return body
}

type server struct {
	network    *slot
	file       *slot
	apiKey     string
	httpClient *http.Client
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
	body := s.network.status()
	body["status"] = "ok"
	body["model_loaded"] = body["loaded"]
	delete(body, "loaded")
	body["file_model"] = s.file.status()
	c.JSON(http.StatusOK, body)
}

func (s *server) reload(c *gin.Context) {
	if !s.requireAuth(c) {
		return
	}
	changed, info, err := s.network.reloadIfChanged(s.httpClient)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"detail": err.Error()})
		return
	}
	body := gin.H{"changed": changed}
	if info != nil {
		body["version"] = info.Version
		body["sha256"] = info.SHA256
	}
	if s.file.manifestURL != "" {
		fileChanged, fileInfo, fileErr := s.file.reloadIfChanged(s.httpClient)
		if fileErr != nil {
			c.JSON(http.StatusBadGateway, gin.H{"detail": fileErr.Error(), "network": body})
			return
		}
		fileBody := gin.H{"changed": fileChanged}
		if fileInfo != nil {
			fileBody["version"] = fileInfo.Version
			fileBody["sha256"] = fileInfo.SHA256
		}
		body["file_model"] = fileBody
	}
	c.JSON(http.StatusOK, body)
}

func (s *server) analyzeNetwork(c *gin.Context) {
	if !s.requireAuth(c) {
		return
	}
	m := s.network.model.Load()
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

	features := m.encodeRequest(payload)
	attackProbability := m.predict(features)
	verdict := bucketVerdict(attackProbability)

	explanation := m.explain(features)
	contributions := rankContributions(explanation.Phi, m.mergedFeatures(features))

	c.JSON(http.StatusOK, gin.H{
		"verdict":      verdict,
		"safety_score": math.Round((1.0-attackProbability)*10000) / 10000,
		"reasoning":    reasoningText(verdict, contributions),
	})
}

func main() {
	networkLocal := ""
	if os.Getenv("NOXOS_MANIFEST_URL") == "" {
		networkLocal = envOr("NOXOS_MODEL_PATH", "model.json")
	}
	s := &server{
		network:    &slot{name: "network", manifestURL: os.Getenv("NOXOS_MANIFEST_URL"), localPath: networkLocal},
		file:       &slot{name: "file", manifestURL: os.Getenv("NOXOS_FILE_MANIFEST_URL"), localPath: os.Getenv("NOXOS_FILE_MODEL_PATH")},
		apiKey:     os.Getenv("NOXOS_INFERENCE_API_KEY"),
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}

	s.network.loadInitial(s.httpClient)
	if s.file.configured() {
		s.file.loadInitial(s.httpClient)
	}

	if intervalSeconds, err := strconv.Atoi(envOr("NOXOS_RELOAD_INTERVAL_SECONDS", "300")); err == nil && intervalSeconds > 0 {
		interval := time.Duration(intervalSeconds) * time.Second
		s.network.startPeriodicReload(s.httpClient, interval)
		s.file.startPeriodicReload(s.httpClient, interval)
	}

	router := s.router()

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
