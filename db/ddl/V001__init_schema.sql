-- Sage 数据库初始化 DDL
-- PostgreSQL 16+

-- ============================================================
-- Schema
-- ============================================================
CREATE SCHEMA IF NOT EXISTS market;
CREATE SCHEMA IF NOT EXISTS fundamental;
CREATE SCHEMA IF NOT EXISTS factor;
CREATE SCHEMA IF NOT EXISTS macro;
CREATE SCHEMA IF NOT EXISTS flow;
CREATE SCHEMA IF NOT EXISTS concept;
CREATE SCHEMA IF NOT EXISTS policy;
CREATE SCHEMA IF NOT EXISTS meta;

-- ============================================================
-- market: 行情数据
-- ============================================================

-- 日线K线（按年分区）
CREATE TABLE market.daily_kline (
    ts_code      VARCHAR(12)    NOT NULL,
    trade_date   DATE           NOT NULL,
    open         NUMERIC(20,4),
    high         NUMERIC(20,4),
    low          NUMERIC(20,4),
    close        NUMERIC(20,4),
    pre_close    NUMERIC(12,4),
    change       NUMERIC(12,4),
    pct_chg      NUMERIC(10,4),
    vol          NUMERIC(18,2),
    amount       NUMERIC(18,4),
    PRIMARY KEY (ts_code, trade_date)
) PARTITION BY RANGE (trade_date);

-- 日线基本面指标（按年分区）
CREATE TABLE market.daily_basic (
    ts_code        VARCHAR(12)  NOT NULL,
    trade_date     DATE         NOT NULL,
    close          NUMERIC(12,4),
    turnover_rate  NUMERIC(10,4),
    turnover_rate_f NUMERIC(10,4),
    volume_ratio   NUMERIC(10,4),
    pe             NUMERIC(14,4),
    pe_ttm         NUMERIC(14,4),
    pb             NUMERIC(14,4),
    ps             NUMERIC(14,4),
    ps_ttm         NUMERIC(14,4),
    dv_ratio       NUMERIC(10,4),
    dv_ttm         NUMERIC(10,4),
    total_share    NUMERIC(18,4),
    float_share    NUMERIC(18,4),
    free_share     NUMERIC(18,4),
    total_mv       NUMERIC(18,4),
    circ_mv        NUMERIC(18,4),
    PRIMARY KEY (ts_code, trade_date)
) PARTITION BY RANGE (trade_date);

-- 指数日线
CREATE TABLE market.index_ohlc (
    ts_code      VARCHAR(12)  NOT NULL,
    trade_date   DATE         NOT NULL,
    open         NUMERIC(20,4),
    high         NUMERIC(20,4),
    low          NUMERIC(20,4),
    close        NUMERIC(20,4),
    pre_close    NUMERIC(12,4),
    change       NUMERIC(12,4),
    pct_chg      NUMERIC(10,4),
    vol          NUMERIC(18,2),
    amount       NUMERIC(18,4),
    PRIMARY KEY (ts_code, trade_date)
);

-- 沪深300成分股权重
CREATE TABLE market.hs300_constituents (
    index_code   VARCHAR(12)  NOT NULL,
    con_code     VARCHAR(12)  NOT NULL,
    trade_date   DATE         NOT NULL,
    weight       NUMERIC(10,6),
    PRIMARY KEY (con_code, trade_date)
);

-- ============================================================
-- fundamental: 财务数据
-- ============================================================

-- 利润表
CREATE TABLE fundamental.income (
    ts_code       VARCHAR(12)  NOT NULL,
    ann_date      DATE,
    f_ann_date    DATE,
    end_date      DATE         NOT NULL,
    report_type   VARCHAR(4)   NOT NULL DEFAULT '1',
    comp_type     VARCHAR(4),
    basic_eps     NUMERIC(12,4),
    diluted_eps   NUMERIC(12,4),
    total_revenue NUMERIC(18,4),
    revenue       NUMERIC(18,4),
    total_cogs    NUMERIC(18,4),
    oper_cost     NUMERIC(18,4),
    sell_exp      NUMERIC(18,4),
    admin_exp     NUMERIC(18,4),
    rd_exp        NUMERIC(18,4),
    fin_exp       NUMERIC(18,4),
    operate_profit NUMERIC(18,4),
    non_oper_income NUMERIC(18,4),
    non_oper_exp  NUMERIC(18,4),
    n_income      NUMERIC(18,4),
    n_income_attr_p NUMERIC(18,4),
    ebit          NUMERIC(18,4),
    ebitda        NUMERIC(18,4),
    PRIMARY KEY (ts_code, end_date, report_type)
);

-- 资产负债表
CREATE TABLE fundamental.balancesheet (
    ts_code       VARCHAR(12)  NOT NULL,
    ann_date      DATE,
    f_ann_date    DATE,
    end_date      DATE         NOT NULL,
    report_type   VARCHAR(4)   NOT NULL DEFAULT '1',
    comp_type     VARCHAR(4),
    total_assets  NUMERIC(18,4),
    total_liab    NUMERIC(18,4),
    total_hldr_eqy_exc_min_int NUMERIC(18,4),
    total_cur_assets  NUMERIC(18,4),
    total_nca     NUMERIC(18,4),
    total_cur_liab NUMERIC(18,4),
    total_ncl     NUMERIC(18,4),
    money_cap     NUMERIC(18,4),
    inventories   NUMERIC(18,4),
    accounts_receiv NUMERIC(18,4),
    fix_assets    NUMERIC(18,4),
    intan_assets  NUMERIC(18,4),
    goodwill      NUMERIC(18,4),
    lt_borr       NUMERIC(18,4),
    st_borr       NUMERIC(18,4),
    PRIMARY KEY (ts_code, end_date, report_type)
);

-- 现金流量表
CREATE TABLE fundamental.cashflow (
    ts_code       VARCHAR(12)  NOT NULL,
    ann_date      DATE,
    f_ann_date    DATE,
    end_date      DATE         NOT NULL,
    report_type   VARCHAR(4)   NOT NULL DEFAULT '1',
    comp_type     VARCHAR(4),
    n_cashflow_act NUMERIC(18,4),
    n_cashflow_inv_act NUMERIC(18,4),
    n_cash_flows_fnc_act NUMERIC(18,4),
    c_fr_sale_sg  NUMERIC(18,4),
    c_paid_goods_s NUMERIC(18,4),
    c_paid_to_for_empl NUMERIC(18,4),
    c_pay_acq_const_fiolta NUMERIC(18,4),
    free_cashflow NUMERIC(18,4),
    PRIMARY KEY (ts_code, end_date, report_type)
);

-- 财务指标
CREATE TABLE fundamental.fina_indicator (
    ts_code       VARCHAR(12)  NOT NULL,
    ann_date      DATE,
    end_date      DATE         NOT NULL,
    roe           NUMERIC(12,4),
    roe_dt        NUMERIC(12,4),
    roa           NUMERIC(12,4),
    grossprofit_margin NUMERIC(12,4),
    netprofit_margin   NUMERIC(12,4),
    debt_to_assets     NUMERIC(12,4),
    current_ratio      NUMERIC(12,4),
    quick_ratio        NUMERIC(12,4),
    eps            NUMERIC(12,4),
    bps            NUMERIC(12,4),
    cfps           NUMERIC(12,4),
    ocfps          NUMERIC(12,4),
    or_yoy         NUMERIC(20,6),
    op_yoy         NUMERIC(20,6),
    netprofit_yoy  NUMERIC(20,6),
    PRIMARY KEY (ts_code, end_date)
);

-- 分红送股
CREATE TABLE fundamental.dividend (
    ts_code       VARCHAR(12)  NOT NULL,
    end_date      DATE         NOT NULL,
    ann_date      DATE,
    div_proc      VARCHAR(20)  NOT NULL DEFAULT '',
    stk_div       NUMERIC(12,6),
    stk_bo_rate   NUMERIC(12,6),
    stk_co_rate   NUMERIC(12,6),
    cash_div      NUMERIC(12,6),
    cash_div_tax  NUMERIC(12,6),
    record_date   DATE,
    ex_date       DATE,
    pay_date      DATE,
    div_listdate  DATE,
    PRIMARY KEY (ts_code, end_date, div_proc)
);

-- ============================================================
-- flow: 资金流向
-- ============================================================

-- 北向资金流向
CREATE TABLE flow.northbound_flow (
    trade_date     DATE PRIMARY KEY,
    ggt_ss         NUMERIC(18,4),
    ggt_sz         NUMERIC(18,4),
    hgt            NUMERIC(18,4),
    sgt            NUMERIC(18,4),
    north_money    NUMERIC(18,4),
    south_money    NUMERIC(18,4)
);

-- 北向持股
CREATE TABLE flow.northbound_hold (
    ts_code      VARCHAR(12)  NOT NULL,
    trade_date   DATE         NOT NULL,
    name         VARCHAR(100),
    vol          NUMERIC(18,2),
    ratio        NUMERIC(10,4),
    exchange     VARCHAR(10),
    PRIMARY KEY (ts_code, trade_date)
);

-- 北向Top10
CREATE TABLE flow.northbound_top10 (
    trade_date   DATE         NOT NULL,
    ts_code      VARCHAR(12)  NOT NULL,
    name         VARCHAR(100),
    close        NUMERIC(12,4),
    change       NUMERIC(12,4),
    rank         INTEGER,
    market_type  VARCHAR(10),
    amount       NUMERIC(18,4),
    net_amount   NUMERIC(18,4),
    buy          NUMERIC(18,4),
    sell         NUMERIC(18,4),
    PRIMARY KEY (trade_date, ts_code)
);

-- 融资融券
CREATE TABLE flow.margin (
    trade_date   DATE         NOT NULL,
    exchange_id  VARCHAR(10)  NOT NULL,
    rzye         NUMERIC(18,4),
    rzmre        NUMERIC(18,4),
    rzche        NUMERIC(18,4),
    rqye         NUMERIC(18,4),
    rqmcl        NUMERIC(18,4),
    rzrqye       NUMERIC(18,4),
    rqylje       NUMERIC(18,4),
    PRIMARY KEY (trade_date, exchange_id)
);

-- 融资融券明细
CREATE TABLE flow.margin_detail (
    trade_date DATE        NOT NULL,
    ts_code    VARCHAR(12) NOT NULL,
    rzye       NUMERIC(20,4),
    rqye       NUMERIC(20,4),
    rzmre      NUMERIC(20,4),
    rqyl       NUMERIC(20,4),
    rzche      NUMERIC(20,4),
    rqchl      NUMERIC(20,4),
    rqmcl      NUMERIC(20,4),
    rzrqye     NUMERIC(20,4),
    PRIMARY KEY (trade_date, ts_code)
);

-- 个股资金流向（按年分区）
CREATE TABLE flow.moneyflow (
    ts_code        VARCHAR(12)  NOT NULL,
    trade_date     DATE         NOT NULL,
    buy_sm_vol     NUMERIC(18,2),
    buy_sm_amount  NUMERIC(18,4),
    sell_sm_vol    NUMERIC(18,2),
    sell_sm_amount NUMERIC(18,4),
    buy_md_vol     NUMERIC(18,2),
    buy_md_amount  NUMERIC(18,4),
    sell_md_vol    NUMERIC(18,2),
    sell_md_amount NUMERIC(18,4),
    buy_lg_vol     NUMERIC(18,2),
    buy_lg_amount  NUMERIC(18,4),
    sell_lg_vol    NUMERIC(18,2),
    sell_lg_amount NUMERIC(18,4),
    buy_elg_vol    NUMERIC(18,2),
    buy_elg_amount NUMERIC(18,4),
    sell_elg_vol   NUMERIC(18,2),
    sell_elg_amount NUMERIC(18,4),
    net_mf_vol     NUMERIC(18,2),
    net_mf_amount  NUMERIC(18,4),
    PRIMARY KEY (ts_code, trade_date)
) PARTITION BY RANGE (trade_date);

-- ============================================================
-- concept: 概念与行业
-- ============================================================

-- 同花顺概念指数
CREATE TABLE concept.ths_index (
    ts_code    VARCHAR(20)  PRIMARY KEY,
    name       VARCHAR(100),
    count      INTEGER,
    exchange   VARCHAR(10),
    list_date  DATE,
    type       VARCHAR(10)
);

-- 概念日线
CREATE TABLE concept.ths_daily (
    ts_code      VARCHAR(20)  NOT NULL,
    trade_date   DATE         NOT NULL,
    open         NUMERIC(20,4),
    high         NUMERIC(20,4),
    low          NUMERIC(20,4),
    close        NUMERIC(20,4),
    pct_change   NUMERIC(20,4),
    vol          NUMERIC(18,2),
    turnover_rate NUMERIC(10,4),
    total_mv     NUMERIC(18,4),
    float_mv     NUMERIC(18,4),
    pe           NUMERIC(14,4),
    PRIMARY KEY (ts_code, trade_date)
);

-- 概念成分股
CREATE TABLE concept.ths_member (
    ts_code    VARCHAR(20)  NOT NULL,
    con_code   VARCHAR(12)  NOT NULL,
    name       VARCHAR(100),
    PRIMARY KEY (ts_code, con_code)
);

-- 申万行业分类
CREATE TABLE concept.sw_industry (
    index_code    VARCHAR(20)  PRIMARY KEY,
    industry_name VARCHAR(100),
    level         VARCHAR(4),
    src           VARCHAR(10),
    list_date     DATE,
    fullname      VARCHAR(200)
);

-- 行业成分股
CREATE TABLE concept.sw_index_member (
    index_code   VARCHAR(20)  NOT NULL,
    index_name   VARCHAR(100),
    con_code     VARCHAR(12)  NOT NULL,
    con_name     VARCHAR(100),
    in_date      DATE,
    out_date     DATE,
    PRIMARY KEY (index_code, con_code)
);

-- ============================================================
-- macro: 宏观数据
-- ============================================================

-- 申万行业估值
CREATE TABLE macro.sw_valuation (
    trade_date    DATE         NOT NULL,
    index_code    VARCHAR(20)  NOT NULL,
    name          VARCHAR(100),
    pe            NUMERIC(14,4),
    pb            NUMERIC(14,4),
    float_share   NUMERIC(18,4),
    total_mv      NUMERIC(18,4),
    float_mv      NUMERIC(18,4),
    turnover_rate NUMERIC(10,4),
    pct_change    NUMERIC(10,4),
    vol           NUMERIC(18,2),
    amount        NUMERIC(18,4),
    PRIMARY KEY (trade_date, index_code)
);

-- 国债收益率
CREATE TABLE macro.yield_curve (
    trade_date   DATE         NOT NULL,
    curve_type   VARCHAR(20)  NOT NULL,
    curve_term   NUMERIC(6,2) NOT NULL,
    yield        NUMERIC(10,6),
    PRIMARY KEY (trade_date, curve_type, curve_term)
);

-- 宏观经济指标（CPI/PPI/PMI 统一表）
CREATE TABLE macro.cn_macro (
    month        VARCHAR(10)  NOT NULL,
    indicator    VARCHAR(20)  NOT NULL,
    nt_val       NUMERIC(12,4),
    nt_yoy       NUMERIC(10,4),
    nt_mom       NUMERIC(10,4),
    nt_accu      NUMERIC(12,4),
    PRIMARY KEY (month, indicator)
);

-- ============================================================
-- policy: 政策/研报元数据（不入库原文，仅保存链接与关键字段）
-- ============================================================

CREATE TABLE policy.gov_notice (
    publish_date DATE         NOT NULL,
    title        TEXT         NOT NULL,
    url          TEXT         NOT NULL,
    source_name  VARCHAR(100),
    source_tag   VARCHAR(100),
    source_type  VARCHAR(50),
    fetched_at   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (publish_date, url)
);

CREATE TABLE policy.industry_report (
    publish_date  DATE         NOT NULL,
    title         TEXT         NOT NULL,
    industry      VARCHAR(100),
    org           VARCHAR(100),
    rating        VARCHAR(50),
    rating_change VARCHAR(50),
    info_code     VARCHAR(50),
    source_name   VARCHAR(100),
    source_type   VARCHAR(50),
    source_url    TEXT,
    fetched_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (publish_date, info_code, title)
);

-- ============================================================
-- meta: 元数据
-- ============================================================

-- 数据同步日志
CREATE TABLE meta.data_sync_log (
    id           SERIAL PRIMARY KEY,
    task_name    VARCHAR(50)  NOT NULL,
    start_date   DATE,
    end_date     DATE,
    record_count INTEGER,
    status       VARCHAR(20),
    error_msg    TEXT,
    created_at   TIMESTAMP DEFAULT NOW()
);

-- 运行配置（最小配置表）
CREATE TABLE meta.app_config (
    code        VARCHAR(100) NOT NULL PRIMARY KEY,
    value       TEXT         NOT NULL,
    is_active   BOOLEAN      NOT NULL DEFAULT TRUE,
    updated_at  TIMESTAMP    DEFAULT NOW(),
    remark      TEXT
);

-- 作业运行日志（周度/日度任务）
CREATE TABLE meta.job_run_log (
    id          SERIAL PRIMARY KEY,
    job_name    VARCHAR(100) NOT NULL,
    run_date    DATE         DEFAULT CURRENT_DATE,
    status      VARCHAR(20)  NOT NULL,
    started_at  TIMESTAMP    DEFAULT NOW(),
    ended_at    TIMESTAMP,
    output_path TEXT,
    message     TEXT,
    created_at  TIMESTAMP    DEFAULT NOW()
);

-- 信号快照登记（输出文件与行数）
CREATE TABLE meta.signal_snapshot (
    signal_name  VARCHAR(100) NOT NULL,
    trade_date   DATE         NOT NULL,
    record_count INTEGER,
    output_path  TEXT,
    created_at   TIMESTAMP    DEFAULT NOW(),
    PRIMARY KEY (signal_name, trade_date)
);

-- ============================================================
-- 分区表：按年创建分区
-- ============================================================

DO $$
DECLARE
    y INT;
    tbl TEXT;
    schema_name TEXT;
BEGIN
    FOR y IN 2016..2026 LOOP
        FOREACH tbl IN ARRAY ARRAY[
            'market.daily_kline',
            'market.daily_basic',
            'flow.moneyflow'
        ] LOOP
            schema_name := split_part(tbl, '.', 1);
            EXECUTE format(
                'CREATE TABLE IF NOT EXISTS %s_%s PARTITION OF %s FOR VALUES FROM (%L) TO (%L)',
                tbl, y, tbl,
                format('%s-01-01', y),
                format('%s-01-01', y + 1)
            );
        END LOOP;
    END LOOP;
END $$;

-- ============================================================
-- 索引
-- ============================================================

CREATE INDEX idx_daily_kline_ts_code ON market.daily_kline (ts_code);
CREATE INDEX idx_daily_kline_trade_date ON market.daily_kline (trade_date);
CREATE INDEX idx_daily_basic_ts_code ON market.daily_basic (ts_code);
CREATE INDEX idx_daily_basic_trade_date ON market.daily_basic (trade_date);
CREATE INDEX idx_income_ann_date ON fundamental.income (ann_date);
CREATE INDEX idx_balancesheet_ann_date ON fundamental.balancesheet (ann_date);
CREATE INDEX idx_fina_indicator_ann_date ON fundamental.fina_indicator (ann_date);
CREATE INDEX idx_moneyflow_ts_code ON flow.moneyflow (ts_code);
CREATE INDEX idx_moneyflow_trade_date ON flow.moneyflow (trade_date);
CREATE INDEX idx_sw_valuation_index_code ON macro.sw_valuation (index_code);
CREATE INDEX idx_hs300_constituents_trade_date ON market.hs300_constituents (trade_date);
