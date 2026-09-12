# Deploying the inference box

Single small EC2 instance, `us-east-1`, account `139229021586` (same account/region as `noxos-os`'s compile fleet). No fleet, no spot-interruption handling, no EBS snapshot dance — that machinery exists in `noxos-os` because AOSP compiles run for hours and can't be restarted cheaply. This box just serves a small gradient-boosted-tree model and ClamAV; if it's interrupted, a fresh instance boots the same systemd unit off the same public repo and is back up in under two minutes with nothing to lose. `t3.micro` on-demand (~$7.50/mo) is plenty for the traffic this gets — the app only ever asks about a destination once per its lifetime (see `AclRepository.nextAnalysisBatch` cost-gating in `noxos-app`), so this is not a high-QPS service.

## IAM: deliberately none

The box makes zero AWS API calls — it `git clone`s this public repo over HTTPS and runs `uvicorn` + `clamd` locally. No S3, no metadata calls beyond nothing needed. Per this project's own rule ("always create a dedicated least-privilege role, scoped to only what it needs, probably nothing beyond basic EC2 networking" — see the briefing), the least-privilege answer here is **no instance profile at all**. If a future version pulls training data or model updates from S3, scope a read-only role to that one bucket then, following the `noxos-server-ci` pattern (dedicated, read-only, single-bucket) — don't reuse `noxos-rom-compile-role`.

This also means none of the IAM-write commands that needed manual execution for the compile box (blocked by this environment's auto-mode classifier) apply here — everything below is a plain EC2 action.

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

**Known gap, flagged not hidden**: port 8443 is plain HTTP behind a shared-secret bearer token (`NOXOS_INFERENCE_API_KEY`), not TLS — there's no domain name yet to get a real cert against, and self-signing would just push the trust problem into the Android client (custom `TrustManager`/cert pinning) for a v0 that doesn't have a live app caller yet either. Same "drafted, not deployed, revisit before it matters" posture as `noxos-server`'s presigned-URL Lambda. Add TLS (Let's Encrypt via a real domain, or a self-signed pin) before this endpoint carries real traffic — token-over-plaintext is not acceptable long-term for a security-focused OS's own backend.

The admin IP restriction on SSH is tighter than the compile box's existing `sg-0092b73a75250a2ed` (open on 0.0.0.0/0) — not fixing that one retroactively here, just not repeating it.

## 2. Launch

```bash
AMI_ID=$(aws ec2 describe-images --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
            "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)

# Edit infra/user-data.sh first: replace CHANGE_ME_BEFORE_LAUNCH with a real
# random token (`openssl rand -hex 32`) — this is what noxos-app's dispatcher
# settings screen needs configured to match.

aws ec2 run-instances --image-id "$AMI_ID" --instance-type t3.micro \
  --security-group-ids "$SG_ID" \
  --user-data file://infra/user-data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=noxos-inference}]' \
  --query 'Instances[0].InstanceId' --output text
```

No SSH key pair specified above — add `--key-name <your-key>` if you want shell access; the box doesn't need it to run (systemd handles restarts).

## 3. Verify

```bash
IP=$(aws ec2 describe-instances --instance-ids <id> --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
curl -s http://$IP:8443/health
curl -s -X POST http://$IP:8443/analyze/network \
  -H "Authorization: Bearer <the token you set>" -H 'Content-Type: application/json' \
  -d '{"ip":"1.1.1.1","port":443,"protocol":"tcp","volume_bytes":1024,"frequency":1,"requesting_app":"com.example"}'
```

## Cost

`t3.micro` on-demand in `us-east-1`: ~$0.0104/hr ≈ $7.50/mo, or free under the 12-month Free Tier if this AWS account is still within that window (`aws ec2 describe-account-attributes` doesn't expose this — check the Billing console). No EBS beyond the default 8GB gp3 root volume (~$0.64/mo). Not using spot: the savings (~70%) aren't worth the interruption/restart churn for an always-on API endpoint, unlike the compile box where spot savings are large in absolute terms and the workload already tolerates interruption by design.

## Not done here, not yet needed

- No autoscaling / load balancer — one instance is enough at this traffic level; revisit if `nextAnalysisBatch`'s batch size or call frequency ever grows enough to matter.
- No CloudWatch alarms/monitoring — `systemctl status noxos-inference` over SSH is enough for a solo project at this stage.
