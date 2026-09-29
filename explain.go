package main

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

const rootPathFeature = ""

type pathElement struct {
	feature      string
	zeroFraction float64
	oneFraction  float64
	pweight      float64
}

func extendPath(path []pathElement, uniqueDepth int, zeroFraction, oneFraction float64, feature string) {
	path[uniqueDepth] = pathElement{feature: feature, zeroFraction: zeroFraction, oneFraction: oneFraction}
	if uniqueDepth == 0 {
		path[uniqueDepth].pweight = 1.0
	}
	for i := uniqueDepth - 1; i >= 0; i-- {
		path[i+1].pweight += oneFraction * path[i].pweight * float64(i+1) / float64(uniqueDepth+1)
		path[i].pweight = zeroFraction * path[i].pweight * float64(uniqueDepth-i) / float64(uniqueDepth+1)
	}
}

func unwindPath(path []pathElement, uniqueDepth, pathIndex int) {
	oneFraction := path[pathIndex].oneFraction
	zeroFraction := path[pathIndex].zeroFraction
	nextOnePortion := path[uniqueDepth].pweight

	for i := uniqueDepth - 1; i >= 0; i-- {
		if oneFraction != 0 {
			tmp := path[i].pweight
			path[i].pweight = nextOnePortion * float64(uniqueDepth+1) / (float64(i+1) * oneFraction)
			nextOnePortion = tmp - path[i].pweight*zeroFraction*float64(uniqueDepth-i)/float64(uniqueDepth+1)
		} else {
			path[i].pweight = (path[i].pweight * float64(uniqueDepth+1)) / (zeroFraction * float64(uniqueDepth-i))
		}
	}

	for i := pathIndex; i < uniqueDepth; i++ {
		path[i].feature = path[i+1].feature
		path[i].zeroFraction = path[i+1].zeroFraction
		path[i].oneFraction = path[i+1].oneFraction
	}
}

func unwoundPathSum(path []pathElement, uniqueDepth, pathIndex int) float64 {
	oneFraction := path[pathIndex].oneFraction
	zeroFraction := path[pathIndex].zeroFraction
	nextOnePortion := path[uniqueDepth].pweight
	total := 0.0

	if oneFraction != 0 {
		for i := uniqueDepth - 1; i >= 0; i-- {
			tmp := nextOnePortion / (float64(i+1) * oneFraction)
			total += tmp
			nextOnePortion = path[i].pweight - tmp*zeroFraction*float64(uniqueDepth-i)
		}
	} else {
		for i := uniqueDepth - 1; i >= 0; i-- {
			total += path[i].pweight / (zeroFraction * float64(uniqueDepth-i))
		}
	}
	return total * float64(uniqueDepth+1)
}

// treeShapRecursive is a direct port of the path-dependent TreeSHAP algorithm
// (Lundberg et al. 2018) from shap's own reference implementation
// (shap/cext/tree_shap.h, tree_shap_recursive), simplified for a single
// output and no interaction-value "condition" parameter, which this project
// never uses.
func treeShapRecursive(n *node, features map[string]float64, phi map[string]float64,
	uniqueDepth int, parentPath []pathElement, parentZeroFraction, parentOneFraction float64, parentFeature string) {

	path := make([]pathElement, uniqueDepth+1)
	if uniqueDepth > 0 {
		copy(path, parentPath[:uniqueDepth])
	}
	extendPath(path, uniqueDepth, parentZeroFraction, parentOneFraction, parentFeature)

	if n.Leaf != nil {
		for i := 1; i <= uniqueDepth; i++ {
			w := unwoundPathSum(path, uniqueDepth, i)
			el := path[i]
			phi[el.feature] += w * (el.oneFraction - el.zeroFraction) * (*n.Leaf)
		}
		return
	}

	value, ok := features[n.Feature]
	goLeft := n.DefaultLeft
	if ok {
		goLeft = float32(value) < float32(n.Threshold)
	}
	hot, cold := n.Right, n.Left
	if goLeft {
		hot, cold = n.Left, n.Right
	}

	w := n.Cover
	hotZeroFraction := hot.Cover / w
	coldZeroFraction := cold.Cover / w

	incomingZeroFraction := 1.0
	incomingOneFraction := 1.0

	pathIndex := -1
	for i := 0; i <= uniqueDepth; i++ {
		if path[i].feature == n.Feature {
			pathIndex = i
			break
		}
	}
	newUniqueDepth := uniqueDepth
	if pathIndex != -1 {
		incomingZeroFraction = path[pathIndex].zeroFraction
		incomingOneFraction = path[pathIndex].oneFraction
		unwindPath(path, newUniqueDepth, pathIndex)
		newUniqueDepth--
	}

	treeShapRecursive(hot, features, phi, newUniqueDepth+1, path, hotZeroFraction*incomingZeroFraction, incomingOneFraction, n.Feature)
	treeShapRecursive(cold, features, phi, newUniqueDepth+1, path, coldZeroFraction*incomingZeroFraction, 0, n.Feature)
}

// computeExpectation mirrors shap's compute_expectations: the cover-weighted
// average of a subtree's leaf values, used as each tree's own contribution
// to the bias/base-value term (its expected output over the training
// population the cover stats were collected from).
func computeExpectation(n *node) float64 {
	if n.Leaf != nil {
		return *n.Leaf
	}
	leftExp := computeExpectation(n.Left)
	rightExp := computeExpectation(n.Right)
	totalCover := n.Left.Cover + n.Right.Cover
	if totalCover == 0 {
		return 0
	}
	return (n.Left.Cover*leftExp + n.Right.Cover*rightExp) / totalCover
}

type explanation struct {
	BaseValue float64
	Phi       map[string]float64
	Margin    float64
}

func (m *model) explain(features map[string]float64) explanation {
	full := m.mergedFeatures(features)

	phi := make(map[string]float64)
	baseValue := m.BaseScore
	for _, t := range m.Trees {
		baseValue += computeExpectation(t)
		treeShapRecursive(t, full, phi, 0, nil, 1.0, 1.0, rootPathFeature)
	}

	margin := baseValue
	for _, v := range phi {
		margin += v
	}

	return explanation{BaseValue: baseValue, Phi: phi, Margin: margin}
}

type featureContribution struct {
	Feature string
	Value   float64
	Phi     float64
}

func rankContributions(phi map[string]float64, features map[string]float64) []featureContribution {
	out := make([]featureContribution, 0, len(phi))
	for f, p := range phi {
		out = append(out, featureContribution{Feature: f, Value: features[f], Phi: p})
	}
	sort.Slice(out, func(i, j int) bool { return math.Abs(out[i].Phi) > math.Abs(out[j].Phi) })
	return out
}

func describeFeatures(cs []featureContribution) string {
	parts := make([]string, len(cs))
	for i, c := range cs {
		parts[i] = fmt.Sprintf("%s (%g)", c.Feature, c.Value)
	}
	return strings.Join(parts, ", ")
}

func reasoningText(verdict string, contributions []featureContribution) string {
	const topN = 3

	take := func(positive bool) []featureContribution {
		var out []featureContribution
		for _, c := range contributions {
			if (positive && c.Phi > 0) || (!positive && c.Phi < 0) {
				out = append(out, c)
			}
			if len(out) == topN {
				break
			}
		}
		return out
	}

	switch verdict {
	case "block":
		drivers := take(true)
		if len(drivers) == 0 {
			return "blocked — no single feature dominated, driven by a broad combination of factors"
		}
		return "blocked — " + describeFeatures(drivers) + " pushed this toward attack-like"
	case "allow":
		drivers := take(false)
		if len(drivers) == 0 {
			return "allowed — traffic matched typical normal patterns closely, nothing stood out"
		}
		return "allowed — " + describeFeatures(drivers) + " matched typical normal traffic"
	default:
		n := topN
		if len(contributions) < n {
			n = len(contributions)
		}
		return "uncertain — " + describeFeatures(contributions[:n]) + " showed mixed signals"
	}
}
