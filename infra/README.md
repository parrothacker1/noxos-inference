# Deploying `teacher-server`

Single small EC2 instance, `us-east-1`, account `139229021586` (same account/region every other box in this project uses). No fleet, no spot, no EBS snapshot dance — same reasoning as the original pre-reset inference box: this serves a small model over a stateless Go binary, cost-gated to one call per newly-flagged destination, not a high-QPS service. `t3.micro` on-demand (~$7.50/mo).

**`/analyze/network` and `/analyze/file` are both served by this one binary.** The file endpoint is a second dynamically loaded model (`NOXOS_FILE_MANIFEST_URL`, release `file-latest`) and is **advisory only**: a Drebin-2012 permission model, unvalidated on modern APKs, that must not gate quarantine or any verdict. `/health` reports both models.

## IAM: deliberately none

Same reasoning as before: the box makes zero AWS API calls — it `git clone`s this public repo over HTTPS, and `teacher-server` itself only calls `github.com`'s public release-download URLs (dynamic model loading, see `TASKS.md`'s "Dynamic model loading" section), no S3, no AWS SDK. No instance profile.

## 1. Security group

```bash
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
MY_IP=$(curl -s https://checkip.amazonaws.com)/32

SG_ID=$(aws ec2 create-security-group --group-name noxos-inference-sg \
  --description "noxos-inference: SSH from admin IP, API from anywhere" \
  --vpc-id "$VPC_ID" --query 'GroupId' --output text)

aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
  --protocol tcp --port 22 --cidr "$MY_IP"

aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
  --protocol tcp --port 8443 --cidr 0.0.0.0/0
```

**Known gap, flagged not hidden, same as before the reset**: port 8443 is plain HTTP behind a shared-secret bearer token, not TLS — no domain name yet. Fix before this carries real traffic.

## 2. Publish the binary, then launch (on-demand, stopped after setup — not left running)

**Do not compile on the box.** The first attempt built `teacher-server` from source on the 1 GiB `t3.micro` and the Go compiler was OOM-killed (`compile: signal: killed` after ~46 minutes of thrashing), so the service never started. Cross-compile locally and publish the static binary instead; `user-data.sh` downloads it and verifies the SHA256:

```bash
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o teacher-server-linux-amd64 .
sha256sum teacher-server-linux-amd64 > teacher-server-linux-amd64.sha256
gh release create teacher-server-latest teacher-server-linux-amd64 teacher-server-linux-amd64.sha256 --target teacher-server
```

The binary release is a manual build, not CI — rebuild and re-publish it whenever `teacher-server`'s code changes.


```bash
AMI_ID=$(aws ec2 describe-images --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
            "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)

# Edit infra/user-data.sh first: replace CHANGE_ME_BEFORE_LAUNCH with a real
# random token (`openssl rand -hex 32`).

INSTANCE_ID=$(aws ec2 run-instances --image-id "$AMI_ID" --instance-type t3.micro \
  --security-group-ids "$SG_ID" \
  --instance-initiated-shutdown-behavior stop \
  --user-data file://infra/user-data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=noxos-inference}]' \
  --query 'Instances[0].InstanceId' --output text)
```

No SSH key pair specified above — add `--key-name <your-key>` for shell access; not required to run.

## 3. Verify, then stop (don't terminate)

```bash
IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
curl -s http://$IP:8443/health
curl -s -X POST http://$IP:8443/analyze/network \
  -H "Authorization: Bearer <the token you set>" -H 'Content-Type: application/json' \
  -d '{"proto":"tcp","dst_port":443}'

aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
```

Warden's config points at this box's inference URL manually (no auto-discovery mechanism, by explicit design) — the IP changes on every start/stop cycle since no Elastic IP is attached; re-check `describe-instances` and update the config each time this gets started for real use, or attach an Elastic IP first if that becomes a recurring friction point (not done here — not asked for).

## Cost

Same as before: ~$7.50/mo on-demand if left running, effectively $0 compute while stopped (the 8GB gp3 root volume still bills, ~$0.64/mo, while stopped).

## Not done here, not yet needed

- No autoscaling / load balancer, no CloudWatch alarms — same reasoning as the pre-reset box.
- No Elastic IP — the instance is stopped between uses, so a static address wasn't asked for.
- `/analyze/file` — no code exists yet, see the note at the top.
