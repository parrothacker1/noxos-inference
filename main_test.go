package main

import (
	"math"
	"testing"
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
