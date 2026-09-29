package main

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

const (
	permissionMarker     = ".permission."
	maxPermissionStrings = 2000
	maxPermissionLength  = 256
	maxFileRequestBytes  = 1 << 20
	fileAdvisoryPrefix   = "advisory only (permission-based model trained on 2012-era apps, do not use this to block) — "
)

func (s *server) router() *gin.Engine {
	router := gin.Default()
	router.GET("/health", s.health)
	router.POST("/reload", s.reload)
	router.POST("/analyze/network", s.analyzeNetwork)
	router.POST("/analyze/file", s.analyzeFile)
	return router
}

func permissionFeatures(m *model, permissionStrings []string) map[string]float64 {
	features := map[string]float64{}
	for _, p := range permissionStrings {
		if len(p) > maxPermissionLength {
			continue
		}
		i := strings.LastIndex(p, permissionMarker)
		if i < 0 {
			continue
		}
		name := p[i+len(permissionMarker):]
		if _, known := m.FeatureDefaults[name]; known {
			features[name] = 1.0
		}
	}
	return features
}

func describePermission(c featureContribution) string {
	if c.Value >= 0.5 {
		return "requests " + c.Feature
	}
	return "does not request " + c.Feature
}

func fileReasoning(verdict string, contributions []featureContribution) string {
	const topN = 3

	pick := func(sign float64) []string {
		var out []string
		for _, c := range contributions {
			if c.Phi*sign > 0 {
				out = append(out, describePermission(c))
			}
			if len(out) == topN {
				break
			}
		}
		return out
	}

	switch verdict {
	case "block":
		if drivers := pick(1); len(drivers) > 0 {
			return fileAdvisoryPrefix + "leans malware-like: " + strings.Join(drivers, ", ")
		}
		return fileAdvisoryPrefix + "leans malware-like with no single permission dominating"
	case "allow":
		if drivers := pick(-1); len(drivers) > 0 {
			return fileAdvisoryPrefix + "leans benign: " + strings.Join(drivers, ", ")
		}
		return fileAdvisoryPrefix + "leans benign with no single permission dominating"
	default:
		n := topN
		if len(contributions) < n {
			n = len(contributions)
		}
		parts := make([]string, n)
		for i := 0; i < n; i++ {
			parts[i] = describePermission(contributions[i])
		}
		return fileAdvisoryPrefix + fmt.Sprintf("mixed signals: %s", strings.Join(parts, ", "))
	}
}

func (s *server) analyzeFile(c *gin.Context) {
	if !s.requireAuth(c) {
		return
	}
	m := s.file.model.Load()
	if m == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"detail": "file model not loaded"})
		return
	}

	var body struct {
		PermissionStrings []string `json:"permission_strings"`
	}
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxFileRequestBytes)
	if c.Request.ContentLength != 0 {
		if err := json.NewDecoder(c.Request.Body).Decode(&body); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"detail": "invalid JSON body"})
			return
		}
	}
	if len(body.PermissionStrings) > maxPermissionStrings {
		c.JSON(http.StatusBadRequest, gin.H{"detail": fmt.Sprintf("permission_strings has more than %d entries", maxPermissionStrings)})
		return
	}

	features := permissionFeatures(m, body.PermissionStrings)
	malwareProbability := m.predict(features)
	verdict := bucketVerdict(malwareProbability)

	explanation := m.explain(features)
	contributions := rankContributions(explanation.Phi, m.mergedFeatures(features))

	c.JSON(http.StatusOK, gin.H{
		"verdict":      verdict,
		"safety_score": math.Round((1.0-malwareProbability)*10000) / 10000,
		"reasoning":    fileReasoning(verdict, contributions),
		"advisory":     true,
	})
}
