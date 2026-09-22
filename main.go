package main

import (
	"encoding/json"
	"log"
	"math"
	"net/http"
	"os"

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

func evalTree(n *node, features map[string]float64) float64 {
	if n.Leaf != nil {
		return *n.Leaf
	}
	value, ok := features[n.Feature]
	goLeft := n.DefaultLeft
	if ok {
		goLeft = value < n.Threshold
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

type server struct {
	model  *model
	apiKey string
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
	c.JSON(http.StatusOK, gin.H{
		"status":       "ok",
		"model_loaded": s.model != nil,
	})
}

func (s *server) analyzeNetwork(c *gin.Context) {
	if !s.requireAuth(c) {
		return
	}
	if s.model == nil {
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

	attackProbability := s.model.predict(s.model.encodeRequest(payload))
	c.JSON(http.StatusOK, gin.H{
		"verdict":      bucketVerdict(attackProbability),
		"safety_score": math.Round((1.0-attackProbability)*10000) / 10000,
		"reasoning":    nil,
	})
}

func main() {
	modelPath := os.Getenv("NOXOS_MODEL_PATH")
	if modelPath == "" {
		modelPath = "model.json"
	}
	m, err := loadModel(modelPath)
	if err != nil {
		log.Printf("model not loaded from %s: %v (starting anyway, /health will report it)", modelPath, err)
	}

	s := &server{model: m, apiKey: os.Getenv("NOXOS_INFERENCE_API_KEY")}
	router := gin.Default()
	router.GET("/health", s.health)
	router.POST("/analyze/network", s.analyzeNetwork)

	addr := ":" + envOr("PORT", "8080")
	log.Printf("listening on %s (model=%s)", addr, modelPath)
	log.Fatal(router.Run(addr))
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
