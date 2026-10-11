package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func syntheticNetworkModel() *model {
	low, high := -3.0, 3.0
	return &model{
		BaseScore: 0,
		FeatureDefaults: map[string]float64{
			"dst_port": 143, "src_byte_count": 1684, "src_packet_count": 14, "dst_byte_count": 2478,
			"dst_packet_count": 18, "duration_millis": 20, "handshake_latency_millis": 0.6,
			"smean": 73, "dmean": 102, "proto": 1,
		},
		Categories: map[string][]string{"proto": {"tcp", "udp"}},
		Trees: []*node{{
			Feature: "src_byte_count", Threshold: 1000000, Cover: 100,
			Left:  &node{Leaf: &low, Cover: 90},
			Right: &node{Leaf: &high, Cover: 10},
		}},
	}
}

func postNetwork(t *testing.T, body string) (int, map[string]any) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	s := &server{network: &slot{name: "network"}, file: &slot{name: "file"}}
	s.network.model.Store(syntheticNetworkModel())
	req := httptest.NewRequest(http.MethodPost, "/analyze/network", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.router().ServeHTTP(rec, req)
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return rec.Code, out
}

func names(v any) []string {
	out := []string{}
	for _, item := range v.([]any) {
		out = append(out, item.(string))
	}
	return out
}

const completeFlow = `"proto":"tcp","dst_port":443,"src_byte_count":30000000,"src_packet_count":21000,"dst_byte_count":9000,"dst_packet_count":100,"duration_millis":20000,"handshake_latency_millis":30`

func TestNetworkRequestWithOnlyIdentityFieldsIsRejected(t *testing.T) {
	code, out := postNetwork(t, `{"ip":"1.2.3.4","priority":"HIGH","reason":"x","first_flagged_epoch_millis":1,"proto":"tcp","dst_port":443}`)
	if code != http.StatusUnprocessableEntity {
		t.Fatalf("code = %d, want 422 (body %v)", code, out)
	}
	if _, hasVerdict := out["verdict"]; hasVerdict {
		t.Errorf("a rejected request must not carry a verdict: %v", out)
	}
	want := []string{"dst_byte_count", "dst_packet_count", "duration_millis", "src_byte_count", "src_packet_count"}
	if got := names(out["missing"]); !reflect.DeepEqual(got, want) {
		t.Errorf("missing = %v, want %v", got, want)
	}
}

func TestNetworkEmptyBodyListsEveryRequiredFeature(t *testing.T) {
	code, out := postNetwork(t, ``)
	if code != http.StatusUnprocessableEntity {
		t.Fatalf("code = %d, want 422", code)
	}
	want := []string{"dst_byte_count", "dst_packet_count", "dst_port", "duration_millis", "proto", "src_byte_count", "src_packet_count"}
	if got := names(out["missing"]); !reflect.DeepEqual(got, want) {
		t.Errorf("missing = %v, want %v", got, want)
	}
}

func TestNetworkSomeMissingNamesOnlyThose(t *testing.T) {
	code, out := postNetwork(t, `{"proto":"udp","dst_port":53,"src_byte_count":10,"src_packet_count":1,"duration_millis":5}`)
	if code != http.StatusUnprocessableEntity {
		t.Fatalf("code = %d, want 422", code)
	}
	want := []string{"dst_byte_count", "dst_packet_count"}
	if got := names(out["missing"]); !reflect.DeepEqual(got, want) {
		t.Errorf("missing = %v, want %v", got, want)
	}
}

func TestNetworkInvalidValuesAreRejected(t *testing.T) {
	code, out := postNetwork(t, `{"proto":"carrier-pigeon","dst_port":"443","src_byte_count":-5,"src_packet_count":1,"dst_byte_count":0,"dst_packet_count":0,"duration_millis":5}`)
	if code != http.StatusUnprocessableEntity {
		t.Fatalf("code = %d, want 422", code)
	}
	want := []string{"dst_port", "proto", "src_byte_count"}
	if got := names(out["invalid"]); !reflect.DeepEqual(got, want) {
		t.Errorf("invalid = %v, want %v", got, want)
	}
}

func TestNetworkCompleteRequestIsScoredAndExtraKeysAreIgnored(t *testing.T) {
	code, out := postNetwork(t, `{`+completeFlow+`,"ip":"1.2.3.4","priority":"HIGH","unknown_future_field":[1,2]}`)
	if code != http.StatusOK {
		t.Fatalf("code = %d, want 200 (body %v)", code, out)
	}
	if out["verdict"] != "block" {
		t.Errorf("a 30 MB upload must reach the tree and block, got %v", out)
	}
}

func TestNetworkMeansAreDerivedAndHandshakeDefaultsToZero(t *testing.T) {
	m := syntheticNetworkModel()
	features, missing, invalid := m.requireFeatures(map[string]any{
		"proto": "udp", "dst_port": 5004.0, "src_byte_count": 2000000.0, "src_packet_count": 1500.0,
		"dst_byte_count": 0.0, "dst_packet_count": 0.0, "duration_millis": 9000.0,
	})
	if len(missing) != 0 || len(invalid) != 0 {
		t.Fatalf("unexpected problems: missing=%v invalid=%v", missing, invalid)
	}
	if features["smean"] != 2000000.0/1500.0 {
		t.Errorf("smean = %v, want bytes/packets", features["smean"])
	}
	if features["dmean"] != 0 {
		t.Errorf("dmean with zero downstream packets = %v, want 0", features["dmean"])
	}
	if features["handshake_latency_millis"] != 0 {
		t.Errorf("handshake_latency_millis = %v, want 0", features["handshake_latency_millis"])
	}
	if features["proto"] != 1 {
		t.Errorf("proto udp = %v, want index 1", features["proto"])
	}
}

func TestNetworkExplicitMeansAreUsedAsSent(t *testing.T) {
	m := syntheticNetworkModel()
	features, missing, invalid := m.requireFeatures(map[string]any{
		"proto": "tcp", "dst_port": 80.0, "src_byte_count": 100.0, "src_packet_count": 10.0,
		"dst_byte_count": 100.0, "dst_packet_count": 10.0, "duration_millis": 1.0, "smean": 77.0, "dmean": 88.0,
	})
	if len(missing) != 0 || len(invalid) != 0 || features["smean"] != 77 || features["dmean"] != 88 {
		t.Errorf("explicit means must pass through: %v missing=%v invalid=%v", features, missing, invalid)
	}
}
