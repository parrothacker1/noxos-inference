package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestPredictAndVerdictBucketing(t *testing.T) {
	leafLow, leafHigh := -2.0, 2.0
	m := &model{
		BaseScore:       0.0,
		FeatureDefaults: map[string]float64{"dst_port": 443.0},
		Categories:      map[string][]string{"proto": {"tcp", "udp"}},
		Trees: []*node{
			{Feature: "dst_port", Threshold: 1024.0, DefaultLeft: true,
				Left:  &node{Leaf: &leafLow},
				Right: &node{Leaf: &leafHigh}},
		},
	}

	allowProba := m.predict(map[string]float64{"dst_port": 80})
	if got := bucketVerdict(allowProba); got != "allow" {
		t.Errorf("dst_port=80: got verdict %q (p=%v), want allow", got, allowProba)
	}

	blockProba := m.predict(map[string]float64{"dst_port": 5000})
	if got := bucketVerdict(blockProba); got != "block" {
		t.Errorf("dst_port=5000: got verdict %q (p=%v), want block", got, blockProba)
	}

	defaultProba := m.predict(map[string]float64{})
	if got := bucketVerdict(defaultProba); got != "allow" {
		t.Errorf("missing dst_port: got verdict %q (p=%v), want allow (default 443 < 1024)", got, defaultProba)
	}

	encoded := m.encodeRequest(map[string]any{"proto": "icmp", "dst_port": 80.0})
	if encoded["proto"] != -1 {
		t.Errorf("unrecognized proto: got %v, want -1", encoded["proto"])
	}
	encoded = m.encodeRequest(map[string]any{"proto": "udp"})
	if encoded["proto"] != 1 {
		t.Errorf("proto=udp: got %v, want 1 (index in categories list)", encoded["proto"])
	}

	if sigmoid(0) != 0.5 {
		t.Errorf("sigmoid(0): got %v, want 0.5", sigmoid(0))
	}
	if math.Abs(sigmoid(1000)-1.0) > 1e-9 {
		t.Errorf("sigmoid(1000): got %v, want ~1.0", sigmoid(1000))
	}
}

func TestReloadIfChangedSwapsModelAndRejectsBadSha256(t *testing.T) {
	modelAJSON := []byte(`{"base_score":0.0,"feature_defaults":{},"trees":[{"leaf":-3.0}]}`)
	modelBJSON := []byte(`{"base_score":0.0,"feature_defaults":{},"trees":[{"leaf":3.0}]}`)
	shaOf := func(data []byte) string {
		sum := sha256.Sum256(data)
		return hex.EncodeToString(sum[:])
	}

	var currentModelBytes []byte
	var manifestSHA string
	mux := http.NewServeMux()
	mux.HandleFunc("/manifest.json", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `{"version":1,"sha256":"%s","modelUrl":"%s/model.json"}`, manifestSHA, "http://"+r.Host)
	})
	mux.HandleFunc("/model.json", func(w http.ResponseWriter, r *http.Request) {
		w.Write(currentModelBytes)
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	s := &server{manifestURL: ts.URL + "/manifest.json", httpClient: &http.Client{Timeout: 5 * time.Second}}

	currentModelBytes = modelAJSON
	manifestSHA = shaOf(modelAJSON)
	changed, info, err := s.reloadIfChanged()
	if err != nil || !changed || info.SHA256 != manifestSHA {
		t.Fatalf("initial load: changed=%v info=%v err=%v", changed, info, err)
	}
	if got := s.currentModel.Load().predict(nil); got != sigmoid(-3.0) {
		t.Errorf("expected modelA's prediction, got %v", got)
	}

	changed, _, err = s.reloadIfChanged()
	if err != nil || changed {
		t.Fatalf("re-fetching the same manifest should report no change: changed=%v err=%v", changed, err)
	}

	currentModelBytes = modelBJSON
	manifestSHA = shaOf(modelBJSON)
	changed, info, err = s.reloadIfChanged()
	if err != nil || !changed || info.SHA256 != manifestSHA {
		t.Fatalf("swap to modelB: changed=%v info=%v err=%v", changed, info, err)
	}
	if got := s.currentModel.Load().predict(nil); got != sigmoid(3.0) {
		t.Errorf("expected modelB's prediction after swap, got %v", got)
	}

	currentModelBytes = modelBJSON
	manifestSHA = "0000000000000000000000000000000000000000000000000000000000000000"
	_, _, err = s.reloadIfChanged()
	if err == nil {
		t.Fatal("expected an error when the manifest's declared sha256 doesn't match the downloaded bytes")
	}
	if got := s.currentModel.Load().predict(nil); got != sigmoid(3.0) {
		t.Errorf("a rejected reload must leave the last-good model in place, got %v", got)
	}
}

func TestEvalTreeUsesFloat32Comparison(t *testing.T) {
	leafLow, leafHigh := -1.0, 1.0
	n := &node{Feature: "duration_millis", Threshold: 0.00100000005,
		Left:  &node{Leaf: &leafLow},
		Right: &node{Leaf: &leafHigh}}

	got := evalTree(n, map[string]float64{"duration_millis": 0.001})
	if got != leafHigh {
		t.Errorf("float64(0.001) < float64(0.00100000005) is true, but float32(0.001) == float32(0.00100000005) "+
			"— XGBoost compares in float32, so this must land right (%v), got %v", leafHigh, got)
	}
}
