#!/usr/bin/env python3
"""Калькулятор юнит-экономики: когортный LTV на валовой прибыли, CAC payback,
LTV/CAC, точка безубыточности, анализ чувствительности, CAC из воронки.

Только стандартная библиотека Python 3.8+.
LTV считается ТОЛЬКО от валовой прибыли и ТОЛЬКО по когорте.
Никаких ARPU × Lifetime и никаких 1/churn.

Примеры:
  # когортный LTV: ARPU $2.4/мес на удержанного, кривая retention, COGS 35%, CAC $12
  python3 ue_calc.py ltv --arpu 2.4 --retention 1,0.55,0.42,0.36,0.32,0.30 \
      --cogs-share 0.35 --cac 12 --periods 12 --sensitivity

  # CAC из воронки закупки
  python3 ue_calc.py funnel --spend 10000 --cpc 0.8 --stages 0.22,0.31 --stage-names "клик→инсталл,инсталл→оплата"

  # сравнение каналов
  python3 ue_calc.py compare --channel "Meta:12:2.4:1,0.55,0.42,0.36" --channel "ASA:22:3.1:1,0.68,0.58,0.52" \
      --cogs-share 0.35 --periods 12
"""
import argparse
import math
import sys

RULE = "-" * 74


def parse_list(s):
    return [float(x) for x in s.replace(" ", "").split(",") if x != ""]


# ---------------------------------------------------------------- ядро модели

def fit_power_tail(retention):
    """Степенная аппроксимация хвоста retention: r(t) = a·t^(-b), t = 1..N.
    Фит по точкам t>=2 методом наименьших квадратов в лог-лог координатах."""
    pts = [(i + 1, r) for i, r in enumerate(retention) if i >= 1 and r > 0]
    if len(pts) < 2:
        return None
    xs = [math.log(t) for t, _ in pts]
    ys = [math.log(r) for _, r in pts]
    n = len(pts)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope * mx
    a, b = math.exp(intercept), -slope
    return a, b


def build_curve(retention, periods):
    """Возвращает (кривая длиной periods, число экстраполированных точек)."""
    curve = list(retention[:periods])
    extrapolated = 0
    if periods > len(curve):
        fit = fit_power_tail(retention)
        if fit is None:
            sys.exit("Для экстраполяции нужно минимум 3 точки retention (или задай --periods ≤ длины кривой).")
        a, b = fit
        for t in range(len(curve) + 1, periods + 1):
            r = min(a * t ** (-b), curve[-1])
            curve.append(max(r, 0.0))
            extrapolated += 1
    return curve, extrapolated


def cohort_table(curve, arpu, cogs_share):
    rows = []
    cum = 0.0
    for i, r in enumerate(curve, 1):
        rev = arpu * r
        gross = rev * (1 - cogs_share)
        cum += gross
        rows.append({"t": i, "retention": r, "revenue": rev, "gross": gross, "cum": cum})
    return rows


def payback_period(rows, cac):
    prev = 0.0
    for row in rows:
        if row["cum"] >= cac:
            need = cac - prev
            step = row["cum"] - prev
            frac = need / step if step else 0.0
            return row["t"] - 1 + frac
        prev = row["cum"]
    return None


def ltv_cac_pair(curve, arpu, cogs_share, cac):
    rows = cohort_table(curve, arpu, cogs_share)
    ltv = rows[-1]["cum"]
    return ltv, (ltv / cac if cac else float("inf")), payback_period(rows, cac), rows


# ---------------------------------------------------------------- команда ltv

def cmd_ltv(a):
    retention = parse_list(a.retention)
    if abs(retention[0] - 1.0) > 1e-9:
        print(f"⚠ Первая точка retention = {retention[0]}, а не 1.0. Убедись, что период 1 — "
              f"это вся когорта, а не «активные на день N». Знаменатель LTV = ИСХОДНЫЙ размер когорты.\n")
    periods = a.periods or len(retention)
    curve, extra = build_curve(retention, periods)
    ltv, ratio, pb, rows = ltv_cac_pair(curve, a.arpu, a.cogs_share, a.cac)
    observed_ltv = cohort_table(curve[:len(retention)], a.arpu, a.cogs_share)[-1]["cum"]
    tail_share = (ltv - observed_ltv) / ltv if ltv else 0.0

    print(RULE)
    print(f"КОГОРТНЫЙ LTV — {a.label}" if a.label else "КОГОРТНЫЙ LTV")
    print(RULE)
    print(f"ARPU на удержанного за период: {a.arpu:.4f}   COGS: {a.cogs_share:.0%}   "
          f"CAC: {a.cac:.4f}   Горизонт: {periods} пер.")
    print()
    print(f"{'Период':>8} | {'Retention':>10} | {'Выручка':>10} | {'Вал.приб.':>10} | {'Кум. LTV':>10} | {'Кум.−CAC':>10}")
    print("-" * 74)
    for row in rows:
        mark = "*" if row["t"] > len(retention) else " "
        print(f"{row['t']:>7}{mark} | {row['retention']:>10.4f} | {row['revenue']:>10.4f} | "
              f"{row['gross']:>10.4f} | {row['cum']:>10.4f} | {row['cum'] - a.cac:>+10.4f}")
    if extra:
        fit = fit_power_tail(retention)
        print(f"\n* {extra} период(ов) экстраполированы степенной моделью r(t) = {fit[0]:.4f}·t^(-{fit[1]:.4f})"
              f" — [ДОПУЩЕНИЕ], не факт.")
        print(f"  Доля LTV из экстраполированного хвоста: {tail_share:.0%}"
              + ("  ⚠ больше трети LTV — это вера в неотнаблюдённое удержание." if tail_share > 0.33 else ""))

    print(RULE)
    print("ИТОГ")
    print(RULE)
    w = 38
    print(f"{f'LTV ({periods} пер., валовая прибыль):':<{w}}{ltv:.4f}")
    print(f"{f'LTV наблюдённый ({len(retention)} пер.):':<{w}}{observed_ltv:.4f}")
    print(f"{'CAC:':<{w}}{a.cac:.4f}")
    print(f"{'LTV / CAC:':<{w}}{ratio:.2f}   (ориентир ≥ 3)")
    print(f"{'CAC payback:':<{w}}" + (f"{pb:.1f} периодов" if pb else
          f"НЕ окупается за {periods} пер. (накоплено {ltv:.4f} из {a.cac:.4f})"))
    print(f"{'Прибыль с юнита за горизонт:':<{w}}{ltv - a.cac:+.4f}")
    print(f"{'ROMI:':<{w}}{((ltv - a.cac) / a.cac if a.cac else 0):+.1%}")
    print(f"{'Макс. допустимый CAC при LTV/CAC=3:':<{w}}{ltv / 3:.4f}")

    verdict = []
    if ratio < 1:
        verdict.append("⛔ Экономика НЕ сходится: юнит в минусе. Масштабирование = ускорение убытка.")
    elif ratio < 3:
        verdict.append(f"⚠ LTV/CAC = {ratio:.2f} < 3: формально плюс, запаса на постоянные расходы нет.")
    else:
        verdict.append(f"✅ LTV/CAC = {ratio:.2f} ≥ 3 на горизонте {periods} пер.")
    if pb and pb > (a.payback_target or 6):
        verdict.append(f"⚠ Payback {pb:.1f} пер. > целевых {a.payback_target or 6}: упрёшься в оборотный "
                       f"капитал раньше, чем в маржу.")
    elif pb:
        verdict.append(f"✅ Payback {pb:.1f} пер. в пределах целевых {a.payback_target or 6}.")
    print()
    for v in verdict:
        print(v)

    if a.sensitivity:
        print()
        print(RULE)
        print("ЧУВСТВИТЕЛЬНОСТЬ: +10% к рычагу → что с LTV/CAC (какой рычаг главный)")
        print(RULE)
        base = ratio
        scenarios = []
        # ARPU +10%
        l, r, p, _ = ltv_cac_pair(curve, a.arpu * 1.1, a.cogs_share, a.cac)
        scenarios.append(("ARPU / чек +10%", l, r, p))
        # retention +10% (кроме периода 1)
        curve_r = [curve[0]] + [min(x * 1.1, 1.0) for x in curve[1:]]
        l, r, p, _ = ltv_cac_pair(curve_r, a.arpu, a.cogs_share, a.cac)
        scenarios.append(("Retention +10%", l, r, p))
        # COGS -10% относительных
        l, r, p, _ = ltv_cac_pair(curve, a.arpu, a.cogs_share * 0.9, a.cac)
        scenarios.append(("COGS −10% (отн.)", l, r, p))
        # CAC -10%
        l, r, p, _ = ltv_cac_pair(curve, a.arpu, a.cogs_share, a.cac * 0.9)
        scenarios.append(("CAC −10%", l, r, p))

        print(f"{'Рычаг':<20} | {'LTV':>10} | {'LTV/CAC':>9} | {'Δ LTV/CAC':>10} | {'Payback':>9}")
        print("-" * 74)
        print(f"{'база':<20} | {ltv:>10.4f} | {base:>9.2f} | {'—':>10} | "
              f"{(f'{pb:.1f}' if pb else 'нет'):>9}")
        for name, l, r, p in scenarios:
            print(f"{name:<20} | {l:>10.4f} | {r:>9.2f} | {r - base:>+10.2f} | "
                  f"{(f'{p:.1f}' if p else 'нет'):>9}")
        best = max(scenarios, key=lambda s: s[2])
        print(f"\nГлавный рычаг: **{best[0]}** → LTV/CAC {best[2]:.2f} ({best[2] - base:+.2f}). "
              f"Туда и прикладывай усилия первым.")

    print()
    print(RULE)
    print("ПРОВЕРЬ ПЕРЕД ТЕМ, КАК ВЕРИТЬ ЭТИМ ЧИСЛАМ:")
    print("  1. ARPU здесь — на УДЕРЖАННОГО пользователя за период; знаменатель LTV — исходная когорта.")
    print("  2. COGS = только переменные: комиссии сторов/эквайринга, инфра, доставка, переменный саппорт.")
    print("     Зарплаты разработки и аренда сюда НЕ входят — это уже операционная прибыль.")
    print("  3. Считай это по КАНАЛУ × СЕГМЕНТУ. Средняя по всем каналам прячет убыточные.")
    print("  4. Для решения «увеличить бюджет» нужна предельная (marginal) экономика: следующая")
    print("     тысяча пользователей дороже предыдущей.")
    print("  5. Инкрементальность: платный канал приписывает себе органику. Пока нет holdout/geo-теста —")
    print("     это [ДОПУЩЕНИЕ], и обычно самое дорогое в модели.")


# ---------------------------------------------------------------- команда funnel

def cmd_funnel(a):
    stages = parse_list(a.stages)
    names = ([s.strip() for s in a.stage_names.split(",")] if a.stage_names
             else [f"этап {i + 1}" for i in range(len(stages))])
    if len(names) != len(stages):
        names = [f"этап {i + 1}" for i in range(len(stages))]

    print(RULE)
    print("CAC ИЗ ВОРОНКИ ЗАКУПКИ")
    print(RULE)
    if a.cpc:
        top = a.spend / a.cpc
        print(f"Бюджет:  {a.spend:,.2f}".replace(",", " "))
        print(f"CPC:     {a.cpc:.4f}  →  кликов: {top:,.0f}".replace(",", " "))
    elif a.cpm and a.ctr:
        impressions = a.spend / a.cpm * 1000
        top = impressions * a.ctr
        print(f"Бюджет:  {a.spend:,.2f}".replace(",", " "))
        print(f"CPM:     {a.cpm:.4f}  →  показов: {impressions:,.0f}".replace(",", " "))
        print(f"CTR:     {a.ctr:.4%}  →  кликов: {top:,.0f}".replace(",", " "))
    else:
        sys.exit("Нужен --cpc либо пара --cpm и --ctr.")

    print()
    print(f"{'Этап':<28} | {'Конверсия':>10} | {'Осталось':>12} | {'Цена шт.':>10}")
    print("-" * 74)
    cur = top
    print(f"{'клики (вход)':<28} | {'—':>10} | {cur:>12,.0f} | {a.spend / cur if cur else 0:>10.4f}"
          .replace(",", " "))
    for name, cr in zip(names, stages):
        cur *= cr
        cpx = a.spend / cur if cur else float("inf")
        print(f"{name:<28} | {cr:>10.2%} | {cur:>12,.0f} | {cpx:>10.4f}".replace(",", " "))

    total_cr = cur / top if top else 0
    print(RULE)
    print(f"Сквозная конверсия клик → цель: {total_cr:.4%}")
    print(f"Целевых юнитов: {cur:,.0f}".replace(",", " "))
    print(f"CAC = {a.spend:,.2f} / {cur:,.0f} = {a.spend / cur if cur else float('inf'):.4f}".replace(",", " "))
    if a.ltv:
        ratio = a.ltv / (a.spend / cur) if cur else 0
        print(f"\nLTV (задан):  {a.ltv:.4f}   →   LTV/CAC = {ratio:.2f}")
        print("✅ сходится" if ratio >= 3 else ("⚠ ниже ориентира 3" if ratio >= 1 else "⛔ юнит в минусе"))
    print("\nКаждую конверсию помечай [ФАКТ] (своя аналитика + дата) или [ДОПУЩЕНИЕ] (вилка low/base/high).")


# ---------------------------------------------------------------- команда compare

def cmd_compare(a):
    print(RULE)
    print("СРАВНЕНИЕ КАНАЛОВ / СЕГМЕНТОВ")
    print(RULE)
    print(f"{'Канал':<16} | {'CAC':>8} | {'ARPU':>7} | {'LTV':>9} | {'LTV/CAC':>8} | {'Payback':>8} | Вердикт")
    print("-" * 92)
    rows = []
    for spec in a.channel:
        parts = spec.split(":")
        if len(parts) != 4:
            sys.exit(f"Формат канала: 'Имя:CAC:ARPU:retention_через_запятую'. Получено: {spec}")
        name, cac, arpu, ret = parts[0], float(parts[1]), float(parts[2]), parse_list(parts[3])
        periods = a.periods or len(ret)
        curve, _ = build_curve(ret, periods)
        ltv, ratio, pb, _ = ltv_cac_pair(curve, arpu, a.cogs_share, cac)
        verdict = "масштабировать" if ratio >= 3 else ("держать/чинить" if ratio >= 1 else "ОТКЛЮЧИТЬ")
        rows.append((name, cac, arpu, ltv, ratio, pb, verdict))
        print(f"{name:<16} | {cac:>8.2f} | {arpu:>7.2f} | {ltv:>9.3f} | {ratio:>8.2f} | "
              f"{(f'{pb:.1f}' if pb else 'нет'):>8} | {verdict}")
    print("-" * 92)
    best = max(rows, key=lambda r: r[4])
    print(f"Лучший по LTV/CAC: {best[0]} ({best[4]:.2f}). Горизонт: {a.periods or 'по длине кривой'} пер.")
    print("Помни: решение «залить бюджет» требует ПРЕДЕЛЬНОЙ экономики канала, а не средней —")
    print("следующая тысяча пользователей дороже предыдущей, и LTV/CAC на масштабе просядет.")


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="Калькулятор юнит-экономики (growth-decision skill)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ltv", help="когортный LTV, payback, LTV/CAC, чувствительность")
    s.add_argument("--arpu", type=float, required=True, help="выручка на УДЕРЖАННОГО пользователя за период")
    s.add_argument("--retention", required=True, help="кривая через запятую, первая точка = 1 (вся когорта)")
    s.add_argument("--cogs-share", type=float, required=True, help="доля переменных затрат в выручке (0.35)")
    s.add_argument("--cac", type=float, required=True)
    s.add_argument("--periods", type=int, help="горизонт; больше длины кривой → экстраполяция")
    s.add_argument("--payback-target", type=float, help="целевой payback в периодах (по умолчанию 6)")
    s.add_argument("--sensitivity", action="store_true", help="анализ чувствительности рычагов")
    s.add_argument("--label", help="подпись (канал/сегмент)")
    s.set_defaults(func=cmd_ltv)

    s = sub.add_parser("funnel", help="CAC из воронки закупки")
    s.add_argument("--spend", type=float, required=True)
    s.add_argument("--cpc", type=float)
    s.add_argument("--cpm", type=float)
    s.add_argument("--ctr", type=float, help="доля, 0.012 = 1.2%%")
    s.add_argument("--stages", required=True, help="конверсии этапов через запятую: 0.22,0.31")
    s.add_argument("--stage-names", help="названия этапов через запятую")
    s.add_argument("--ltv", type=float, help="если известен — посчитает LTV/CAC")
    s.set_defaults(func=cmd_funnel)

    s = sub.add_parser("compare", help="сравнение каналов/сегментов")
    s.add_argument("--channel", action="append", required=True,
                   help="'Имя:CAC:ARPU:retention' — повторить для каждого канала")
    s.add_argument("--cogs-share", type=float, required=True)
    s.add_argument("--periods", type=int)
    s.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
