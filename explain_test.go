package main

import (
	"math"
	"testing"
)

func syntheticExplainModel() *model {
	leafA, leafB, leafC, leafD := -1.5, 0.5, -0.2, 2.0
	return &model{
		BaseScore: 0.1,
		Trees: []*node{
			{
				Feature: "dst_port", Threshold: 1024.0, DefaultLeft: true, Cover: 100.0,
				Left: &node{
					Feature: "duration_millis", Threshold: 500.0, DefaultLeft: true, Cover: 60.0,
					Left:  &node{Leaf: &leafA, Cover: 40.0},
					Right: &node{Leaf: &leafB, Cover: 20.0},
				},
				Right: &node{
					Feature: "duration_millis", Threshold: 500.0, DefaultLeft: true, Cover: 40.0,
					Left:  &node{Leaf: &leafC, Cover: 15.0},
					Right: &node{Leaf: &leafD, Cover: 25.0},
				},
			},
		},
	}
}

func TestExplainSHAPEfficiencyAndGoldenValues(t *testing.T) {
	m := syntheticExplainModel()

	e1 := m.explain(map[string]float64{"dst_port": 80.0, "duration_millis": 200.0})
	if math.Abs(e1.BaseValue-0.07) > 1e-9 {
		t.Errorf("case1 base_value: got %v want 0.07", e1.BaseValue)
	}
	if math.Abs(e1.Phi["dst_port"]-(-0.6616666666666666)) > 1e-9 {
		t.Errorf("case1 phi[dst_port]: got %v want -0.6616666666666666", e1.Phi["dst_port"])
	}
	if math.Abs(e1.Phi["duration_millis"]-(-0.8083333333333333)) > 1e-9 {
		t.Errorf("case1 phi[duration_millis]: got %v want -0.8083333333333333", e1.Phi["duration_millis"])
	}

	e2 := m.explain(map[string]float64{"dst_port": 5000.0, "duration_millis": 900.0})
	if math.Abs(e2.BaseValue-0.07) > 1e-9 {
		t.Errorf("case2 base_value: got %v want 0.07", e2.BaseValue)
	}
	if math.Abs(e2.Phi["dst_port"]-1.0525000000000002) > 1e-9 {
		t.Errorf("case2 phi[dst_port]: got %v want 1.0525000000000002", e2.Phi["dst_port"])
	}
	if math.Abs(e2.Phi["duration_millis"]-0.9774999999999999) > 1e-9 {
		t.Errorf("case2 phi[duration_millis]: got %v want 0.9774999999999999", e2.Phi["duration_millis"])
	}

	for _, tc := range []struct {
		features map[string]float64
		want     float64
	}{
		{map[string]float64{"dst_port": 80.0, "duration_millis": 200.0}, -1.4},
		{map[string]float64{"dst_port": 5000.0, "duration_millis": 900.0}, 2.1},
	} {
		e := m.explain(tc.features)
		margin := e.BaseValue
		for _, p := range e.Phi {
			margin += p
		}
		if math.Abs(margin-tc.want) > 1e-9 {
			t.Errorf("shap efficiency property (base + sum(phi) == real margin) violated: got %v want %v", margin, tc.want)
		}
	}
}

func TestReasoningTextNamesRealDrivers(t *testing.T) {
	contributions := []featureContribution{
		{Feature: "dst_port", Value: 5000, Phi: 1.05},
		{Feature: "duration_millis", Value: 900, Phi: 0.98},
	}
	blockText := reasoningText("block", contributions)
	if blockText == "" {
		t.Fatal("expected non-empty reasoning text for block")
	}

	allowContributions := []featureContribution{
		{Feature: "dst_port", Value: 80, Phi: -0.66},
		{Feature: "duration_millis", Value: 200, Phi: -0.81},
	}
	allowText := reasoningText("allow", allowContributions)
	if allowText == "" {
		t.Fatal("expected non-empty reasoning text for allow")
	}
	if blockText == allowText {
		t.Errorf("block and allow reasoning text should differ, got the same: %q", blockText)
	}
}
