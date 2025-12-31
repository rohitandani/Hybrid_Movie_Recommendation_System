############################
# SNAT AUTOSCALE SCRIPT
############################
cat << 'EOF' > /home/admin/snat-autoscale.sh
#!/bin/bash
set -euo pipefail

PARTITION="/Sample_01"
DG_NAME="device_to_snat_dg"
SNAT_PREFIX="SNAT_"
BASE_NET="10.0.0"        # 10.0.0.0/24 ONLY
IPS_PER_SNAT=3
BLOCK_GAP=10             # spacing inside /24

LOCK="/var/run/snat_autoscale.lock"

exec 200>$LOCK
flock -n 200 || exit 0

HOST=$(hostname)

tmsh -q -c "cd $PARTITION; list ltm data-group internal $DG_NAME" \
|| tmsh -c "cd $PARTITION; create ltm data-group internal $DG_NAME type string"

tmsh -q -c "cd $PARTITION; list ltm data-group internal $DG_NAME records" \
| grep -q "$HOST" && exit 0

EXISTING=$(tmsh -q -c "cd $PARTITION; list ltm snatpool one-line" | awk '{print $3}' | sed 's/SNAT_//')

IDX=1
while echo "$EXISTING" | grep -qx "$IDX"; do IDX=$((IDX+1)); done

SNAT="SNAT_"$IDX
START=$((1 + (IDX - 1) * BLOCK_GAP))

IPS=""
for i in $(seq 0 $((IPS_PER_SNAT - 1))); do
  IPS="$IPS $BASE_NET.$((START + i))"
done

tmsh -c "cd $PARTITION; create ltm snatpool $SNAT members add { $IPS }" || true
tmsh -c "cd $PARTITION; modify ltm data-group internal $DG_NAME records add { $HOST { data $SNAT } }"
tmsh save sys config
EOF

# startup-script
