#!/usr/bin/env bash
set -euo pipefail

mode="${1:-incremental}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
backup_root="${SAGE_DB_BACKUP_ROOT:-/Volumes/SPEED/BizData/Stock/Sage/db}"
container="${SAGE_DB_CONTAINER:-sage-postgres}"
db_name="${SAGE_DB_NAME:-sage_db}"
db_user="${SAGE_DB_USER:-sage}"
db_password="${SAGE_DB_PASSWORD:-sage_dev_2026}"

log_dir="${backup_root}/logs"
mkdir -p "$log_dir"
log_file="${log_dir}/pg_backup_$(date +%Y%m%d).log"
exec > >(tee -a "$log_file") 2>&1

if ! command -v docker >/dev/null 2>&1; then
  echo "docker 未安装或不可用" >&2
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
  echo "Postgres 容器未运行: ${container}" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d)"
full_dir="${backup_root}/full/${timestamp}"
inc_dir="${backup_root}/incremental/${timestamp}"
state_file="${backup_root}/backup_state.txt"
manifest_dir="${backup_root}/manifests"
mkdir -p "$manifest_dir"

run_pg_dump() {
  local args=("$@")
  docker exec -i -e PGPASSWORD="${db_password}" "${container}" pg_dump -U "${db_user}" -d "${db_name}" "${args[@]}"
}

run_psql() {
  local sql="$1"
  docker exec -i -e PGPASSWORD="${db_password}" "${container}" psql -U "${db_user}" -d "${db_name}" -t -A -c "${sql}"
}

get_columns() {
  local schema="$1"
  local table="$2"
  run_psql "SELECT string_agg(quote_ident(column_name), ',') FROM information_schema.columns WHERE table_schema='${schema}' AND table_name='${table}' ORDER BY ordinal_position;"
}

copy_incremental() {
  local schema="$1"
  local table="$2"
  local date_col="$3"
  local start_date="$4"
  local file_path="$5"

  local columns
  columns="$(get_columns "${schema}" "${table}")"
  if [[ -z "${columns}" ]]; then
    echo "跳过 ${schema}.${table}: 无法获取列信息"
    return
  fi

  local where_clause
  if [[ "${date_col}" == "month" ]]; then
    local start_month
    start_month="$(python - <<PY
from datetime import datetime, timedelta
dt = datetime.strptime("${start_date}", "%Y-%m-%d")
print(dt.strftime("%Y%m"))
PY
)"
    where_clause="WHERE ${date_col} >= '${start_month}'"
  else
    where_clause="WHERE ${date_col} >= DATE '${start_date}'"
  fi

  echo "COPY ${schema}.${table} (${columns}) FROM STDIN WITH CSV;" >"${file_path}"
  docker exec -i -e PGPASSWORD="${db_password}" "${container}" psql -U "${db_user}" -d "${db_name}" -c \
    "COPY (SELECT ${columns} FROM ${schema}.${table} ${where_clause}) TO STDOUT WITH CSV" >>"${file_path}"
  echo "\\." >>"${file_path}"
}

if [[ "${mode}" == "full" ]]; then
  mkdir -p "${full_dir}"
  echo "开始全量备份: ${full_dir}"
  run_pg_dump --no-owner --no-privileges --format=plain --verbose | gzip -c >"${full_dir}/sage_full_${timestamp}.sql.gz"
  docker exec -i -e PGPASSWORD="${db_password}" "${container}" pg_dumpall -U "${db_user}" --globals-only \
    | gzip -c >"${full_dir}/globals_${timestamp}.sql.gz"
  echo "${timestamp}" >"${state_file}"
  cat >"${manifest_dir}/full_${timestamp}.json" <<EOF
{"mode":"full","date":"${timestamp}","db":"${db_name}","container":"${container}"}
EOF
  echo "全量备份完成"
  exit 0
fi

if [[ "${mode}" != "incremental" ]]; then
  echo "不支持的模式: ${mode} (仅支持 full/incremental)" >&2
  exit 2
fi

if [[ -f "${state_file}" ]]; then
  last_date="$(cat "${state_file}")"
  start_date="$(python - <<PY
from datetime import datetime, timedelta
dt = datetime.strptime("${last_date}", "%Y%m%d")
print((dt + timedelta(days=1)).strftime("%Y-%m-%d"))
PY
)"
else
  start_date="$(date -v-1d +%Y-%m-%d)"
fi

mkdir -p "${inc_dir}"
echo "开始增量备份: ${inc_dir}, start_date=${start_date}"

tables=(
  "market.daily_kline|trade_date"
  "market.daily_basic|trade_date"
  "market.index_ohlc|trade_date"
  "market.hs300_constituents|trade_date"
  "flow.northbound_flow|trade_date"
  "flow.northbound_hold|trade_date"
  "flow.northbound_top10|trade_date"
  "flow.margin|trade_date"
  "flow.moneyflow|trade_date"
  "concept.ths_daily|trade_date"
  "macro.yield_curve|trade_date"
  "macro.cn_macro|month"
)

for item in "${tables[@]}"; do
  schema_table="${item%%|*}"
  date_col="${item##*|}"
  schema="${schema_table%%.*}"
  table="${schema_table##*.}"
  out_file="${inc_dir}/${schema}.${table}.sql"
  echo "增量导出: ${schema_table}"
  copy_incremental "${schema}" "${table}" "${date_col}" "${start_date}" "${out_file}"
  gzip -f "${out_file}"
done

echo "${timestamp}" >"${state_file}"
cat >"${manifest_dir}/incremental_${timestamp}.json" <<EOF
{"mode":"incremental","date":"${timestamp}","start_date":"${start_date}","db":"${db_name}","container":"${container}"}
EOF

echo "增量备份完成"
