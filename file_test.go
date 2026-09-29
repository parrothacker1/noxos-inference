package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func syntheticFileModel() *model {
	absent, present := -1.0, 4.0
	return &model{
		BaseScore:       -2.0,
		FeatureDefaults: map[string]float64{"SEND_SMS": 0, "INTERNET": 0},
		Trees: []*node{{
			Feature: "SEND_SMS", Threshold: 0.5, DefaultLeft: true, Cover: 100,
			Left:  &node{Leaf: &absent, Cover: 80},
			Right: &node{Leaf: &present, Cover: 20},
		}},
	}
}

func TestPermissionFeaturesMapsOnlyKnownPermissions(t *testing.T) {
	m := syntheticFileModel()
	got := permissionFeatures(m, []string{
		"android.permission.SEND_SMS",
		"android.permission.NOT_IN_VOCABULARY",
		"com.example.custom.permission.INTERNET",
		"no marker at all",
		"android.permission." + strings.Repeat("A", maxPermissionLength),
	})
	if len(got) != 2 || got["SEND_SMS"] != 1 || got["INTERNET"] != 1 {
		t.Errorf("expected exactly SEND_SMS and INTERNET set, got %v", got)
	}
}

func postFile(t *testing.T, s *server, key, body string) (int, map[string]any) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	req := httptest.NewRequest(http.MethodPost, "/analyze/file", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	rec := httptest.NewRecorder()
	s.router().ServeHTTP(rec, req)
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return rec.Code, out
}

func TestAnalyzeFileEndpoint(t *testing.T) {
	s := &server{network: &slot{name: "network"}, file: &slot{name: "file"}, apiKey: "secret"}

	if code, _ := postFile(t, s, "secret", `{}`); code != http.StatusServiceUnavailable {
		t.Errorf("no file model loaded: got %d want 503", code)
	}

	s.file.model.Store(syntheticFileModel())

	if code, _ := postFile(t, s, "", `{}`); code != http.StatusUnauthorized {
		t.Errorf("missing key: got %d want 401", code)
	}
	if code, _ := postFile(t, s, "secret", `not json`); code != http.StatusBadRequest {
		t.Errorf("bad json: got %d want 400", code)
	}

	code, out := postFile(t, s, "secret", `{"permission_strings":["android.permission.SEND_SMS"],"package_name":"com.secret.app","sha256":"abc"}`)
	if code != http.StatusOK || out["verdict"] != "block" || out["advisory"] != true {
		t.Fatalf("SEND_SMS app: code=%d out=%v, want 200 block advisory", code, out)
	}
	reasoning, _ := out["reasoning"].(string)
	if !strings.Contains(reasoning, "advisory only") || !strings.Contains(reasoning, "requests SEND_SMS") {
		t.Errorf("reasoning should be advisory-worded and name the permission, got %q", reasoning)
	}
	if strings.Contains(reasoning, "com.secret.app") {
		t.Errorf("reasoning must never echo the package name, got %q", reasoning)
	}

	code, out = postFile(t, s, "secret", `{"permission_strings":[]}`)
	if code != http.StatusOK || out["verdict"] != "allow" {
		t.Errorf("empty permission list: code=%d out=%v, want 200 allow", code, out)
	}
}
