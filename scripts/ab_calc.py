#!/usr/bin/env python3
"""Калькулятор A/B-тестов: размер выборки, MDE, длительность, оценка результата, SRM.

Только стандартная библиотека Python 3.8+.

Примеры:
  # сколько нужно на группу: baseline 4%, ловим относительный прирост 10%, 12000 юзеров/день
  python3 ab_calc.py size --baseline 0.04 --mde-rel 0.10 --daily 12000

  # что вообще сможем поймать за 14 дней при 12000 юзеров/день
  python3 ab_calc.py mde --baseline 0.04 --daily 12000 --days 14

  # выборка для средней метрики (ARPU)
  python3 ab_calc.py size-mean --mean 1.8 --sd 6.4 --mde-rel 0.05 --daily 12000

  # оценка результата теста по конверсиям
  python3 ab_calc.py eval --n-a 24000 --c-a 962 --n-b 24100 --c-b 1071 --mde-rel 0.10

  # оценка результата по средним
  python3 ab_calc.py eval-mean --n-a 24000 --mean-a 1.80 --sd-a 6.4 --n-b 24000 --mean-b 1.91 --sd-b 6.6

  # проверка перекоса групп
  python3 ab_calc.py srm --n-a 24000 --n-b 23100

  # поправка на множественные сравнения
  python3 ab_calc.py multi --pvalues 0.011,0.032,0.048,0.21 --alpha 0.05
"""
import argparse
import math
import sys
from statistics import NormalDist

ND = NormalDist()
RULE = "-" * 68


def z_of(q: float) -> float:
    return ND.inv_cdf(q)


def two_sided_p(z: float) -> float:
    return 2 * (1 - ND.cdf(abs(z)))


def z_alpha(alpha: float, sides: int) -> float:
    return z_of(1 - alpha / 2) if sides == 2 else z_of(1 - alpha)


def fmt_pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%"


def fmt_p(p: float) -> str:
    return "<1e-12" if p < 1e-12 else f"{p:.2e}"


def thousands(n) -> str:
    return f"{n:,}".replace(",", " ")


def resolve_mde_abs(baseline, mde_abs, mde_rel):
    if mde_abs is None and mde_rel is None:
        sys.exit("Укажи --mde-abs (абсолютный) или --mde-rel (относительный).")
    if mde_abs is not None and mde_rel is not None:
        sys.exit("Укажи что-то одно: --mde-abs или --mde-rel.")
    return mde_abs if mde_abs is not None else baseline * mde_rel


# ---------------------------------------------------------------- размер выборки

def n_per_group_prop(p1, mde_abs, alpha, power, sides):
    p2 = p1 + mde_abs
    if not (0 < p2 < 1):
        sys.exit(f"baseline + MDE = {p2:.4f} вне (0;1). Проверь входные данные.")
    za, zb = z_alpha(alpha, sides), z_of(power)
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / mde_abs ** 2)


def n_per_group_mean(sd, mde_abs, alpha, power, sides):
    za, zb = z_alpha(alpha, sides), z_of(power)
    return math.ceil(2 * (za + zb) ** 2 * sd ** 2 / mde_abs ** 2)


def duration_block(n, daily, groups=2):
    if not daily:
        return ["Длительность: укажи --daily (сколько подходящих пользователей в день), чтобы получить срок."]
    days = groups * n / daily
    out = [
        f"Длительность:            {math.ceil(days)} дн. (формула: {groups}·n / {daily:.0f} в день)",
        f"  → округляя до целых недель: {max(1, math.ceil(days / 7))} нед.",
    ]
    if days > 42:
        out.append("  ⚠ >6 недель: тест почти наверняка съест сезонность и релизы. "
                   "Либо укрупни гипотезу (больше MDE), либо режь экспозицию (только затронутые юзеры), либо CUPED.")
    return out


def cmd_size(a):
    mde = resolve_mde_abs(a.baseline, a.mde_abs, a.mde_rel)
    n = n_per_group_prop(a.baseline, mde, a.alpha, a.power, a.sides)
    print(RULE)
    print("РАЗМЕР ВЫБОРКИ — конверсионная метрика")
    print(RULE)
    print(f"Baseline:                {fmt_pct(a.baseline)}")
    print(f"MDE:                     {fmt_pct(mde)} абс. ({fmt_pct(mde / a.baseline, 1)} отн.) "
          f"→ целевая {fmt_pct(a.baseline + mde)}")
    print(f"alpha={a.alpha}, power={a.power}, критерий {a.sides}-сторонний")
    print(f"n на группу:             {n:,}".replace(",", " "))
    print(f"Всего в тест:            {2 * n:,}".replace(",", " "))
    for line in duration_block(n, a.daily):
        print(line)
    print(RULE)
    half = thousands(n_per_group_prop(a.baseline, mde / 2, a.alpha, a.power, a.sides))
    print(f"Чувствительность: чтобы поймать вдвое меньший эффект ({fmt_pct(mde / 2)} абс.), "
          f"нужно {half} на группу (×4).")
    print("Правило: n растёт как 1/MDE². Не хватает трафика — меняй гипотезу на более крупную, а не α.")
    print("ЗАФИКСИРУЙ n и MDE ДО СТАРТА. Решение по промежуточным данным = подглядывание, вывод недействителен.")


def cmd_size_mean(a):
    mde = resolve_mde_abs(a.mean, a.mde_abs, a.mde_rel)
    n = n_per_group_mean(a.sd, mde, a.alpha, a.power, a.sides)
    cv = a.sd / a.mean if a.mean else float("nan")
    print(RULE)
    print("РАЗМЕР ВЫБОРКИ — метрика-среднее (ARPU, чек, число действий)")
    print(RULE)
    print(f"Среднее (baseline):      {a.mean:.4f}")
    print(f"SD:                      {a.sd:.4f}   (коэф. вариации {cv:.2f})")
    print(f"MDE:                     {mde:.4f} абс. ({fmt_pct(mde / a.mean, 1)} отн.)")
    print(f"alpha={a.alpha}, power={a.power}, критерий {a.sides}-сторонний")
    print(f"n на группу:             {n:,}".replace(",", " "))
    print(f"Всего в тест:            {2 * n:,}".replace(",", " "))
    for line in duration_block(n, a.daily):
        print(line)
    print(RULE)
    if cv > 3:
        print("⚠ Коэф. вариации > 3 — метрика очень шумная (типично для ARPU с китами).")
        print("  Варианты: винзоризация/капирование выбросов, лог-трансформация, CUPED, "
              "или переход на конверсионную прокси-метрику.")


def cmd_mde(a):
    if a.n is None:
        if not (a.daily and a.days):
            sys.exit("Нужен --n на группу либо пара --daily и --days.")
        a.n = int(a.daily * a.days / 2)
    lo, hi = 1e-9, min(a.baseline, 1 - a.baseline) * 0.999
    for _ in range(200):
        mid = (lo + hi) / 2
        if n_per_group_prop(a.baseline, mid, a.alpha, a.power, a.sides) > a.n:
            lo = mid
        else:
            hi = mid
    mde = hi
    print(RULE)
    print("MDE — какой эффект вообще поймаем при заданном трафике")
    print(RULE)
    print(f"n на группу:             {a.n:,}".replace(",", " "))
    print(f"Baseline:                {fmt_pct(a.baseline)}")
    print(f"alpha={a.alpha}, power={a.power}, критерий {a.sides}-сторонний")
    print(f"Минимально ловимый эффект: {fmt_pct(mde)} абс. = {fmt_pct(mde / a.baseline, 1)} отн.")
    print(RULE)
    print("Если бизнесово значимый эффект МЕНЬШЕ этого — тест бессмысленен: «не значимо» будет")
    print("означать «не измерили», а не «не работает». Тогда: копить трафик, резать экспозицию")
    print("до затронутых пользователей, брать более чувствительную метрику или укрупнять гипотезу.")


# ---------------------------------------------------------------- оценка результата

def cmd_eval(a):
    p1, p2 = a.c_a / a.n_a, a.c_b / a.n_b
    diff = p2 - p1
    pooled = (a.c_a + a.c_b) / (a.n_a + a.n_b)
    se_pool = math.sqrt(pooled * (1 - pooled) * (1 / a.n_a + 1 / a.n_b))
    z = diff / se_pool if se_pool else 0.0
    p_value = two_sided_p(z)
    se_unpool = math.sqrt(p1 * (1 - p1) / a.n_a + p2 * (1 - p2) / a.n_b)
    zc = z_alpha(a.alpha, 2)
    ci = (diff - zc * se_unpool, diff + zc * se_unpool)
    rel = diff / p1 if p1 else float("nan")
    ci_rel = (ci[0] / p1, ci[1] / p1) if p1 else (float("nan"),) * 2

    print(RULE)
    print("ОЦЕНКА РЕЗУЛЬТАТА — конверсии")
    print(RULE)
    srm_p = srm_pvalue(a.n_a, a.n_b, a.split)
    if srm_p < 0.001:
        print(f"⛔ SRM: соотношение групп {a.n_a}/{a.n_b} расходится с ожидаемым {a.split:.2f} "
              f"(p={fmt_p(srm_p)}).")
        print("   Рандомизация или логирование сломаны — ЛЮБОЙ вывод ниже недействителен.")
        print("   Сначала чинить сплит и перезапускать тест.\n")
    print(f"A (контроль): {a.c_a:,}/{a.n_a:,} = {fmt_pct(p1)}".replace(",", " "))
    print(f"B (тест):     {a.c_b:,}/{a.n_b:,} = {fmt_pct(p2)}".replace(",", " "))
    print(f"Разница:      {fmt_pct(diff)} абс.  |  {fmt_pct(rel, 1)} отн.")
    print(f"{int((1 - a.alpha) * 100)}% ДИ разницы: [{fmt_pct(ci[0])} ; {fmt_pct(ci[1])}] абс."
          f"   =  [{fmt_pct(ci_rel[0], 1)} ; {fmt_pct(ci_rel[1], 1)}] отн.")
    print(f"z = {z:.3f}   p-value (двусторонний) = {p_value:.4f}")
    print(f"Вывод по значимости: {'значимо' if p_value < a.alpha else 'НЕ значимо'} при alpha={a.alpha}")
    print(RULE)

    warn = []
    if srm_p < 0.001:
        warn.append("SRM (см. выше): результат теста читать нельзя, пока не починен сплит.")
    mde_abs = None
    if a.mde_abs is not None or a.mde_rel is not None:
        mde_abs = resolve_mde_abs(p1, a.mde_abs, a.mde_rel)
        print(f"Заявленный MDE: {fmt_pct(mde_abs)} абс. ({fmt_pct(mde_abs / p1, 1)} отн.)")
        if p_value < a.alpha and diff < mde_abs:
            warn.append("Статзначимо, но эффект МЕНЬШЕ заявленного MDE — практической ценности нет. "
                        "Это отрицательный результат, не победа.")
        if p_value >= a.alpha and ci[1] > mde_abs:
            warn.append("Не значимо, но верхняя граница ДИ выше MDE — мощности не хватило. "
                        "Вывод «не работает» делать нельзя, это «не измерили».")
        if p_value >= a.alpha and ci[1] <= mde_abs:
            print("Не значимо И весь ДИ ниже MDE → эффекта нужного размера нет. Валидный отрицательный результат.")
    else:
        warn.append("Не указан --mde-rel/--mde-abs: без заранее зафиксированного MDE нельзя отличить "
                    "«победу» от статистического шума нужного знака.")

    if ci[0] < 0 < ci[1]:
        print("ДИ включает ноль → направление эффекта не установлено.")

    print()
    print("ОБЯЗАТЕЛЬНЫЕ ПРОВЕРКИ ПЕРЕД РЕШЕНИЕМ:")
    print("  1. n был зафиксирован ДО старта, и тест не останавливали при первом пересечении порога?")
    print("     (2 подглядывания завышают реальный p-value ×2, 5 → ×3.2, автоскрипт → >×12)")
    print("  2. Тест шёл целое число недель и покрыл полный цикл активности?")
    print("  3. Эффект устойчив по дням, а не всплеск первых суток (novelty effect)?")
    print("  4. Guardrail-метрики не просели (crash, скорость, отписки, саппорт)?")
    print("  5. Целевая метрика была одна? Если перебирали метрики/сегменты — нужна поправка (см. `multi`).")
    print("  6. В тест попадали только те, чей опыт реально менялся, и метрики считались с момента экспозиции?")
    for w in warn:
        print(f"\n⚠ {w}")


def cmd_eval_mean(a):
    diff = a.mean_b - a.mean_a
    se = math.sqrt(a.sd_a ** 2 / a.n_a + a.sd_b ** 2 / a.n_b)
    z = diff / se if se else 0.0
    p_value = two_sided_p(z)
    zc = z_alpha(a.alpha, 2)
    ci = (diff - zc * se, diff + zc * se)
    rel = diff / a.mean_a if a.mean_a else float("nan")
    print(RULE)
    print("ОЦЕНКА РЕЗУЛЬТАТА — средние")
    print(RULE)
    print(f"A: n={a.n_a:,}  mean={a.mean_a:.4f}  sd={a.sd_a:.4f}".replace(",", " "))
    print(f"B: n={a.n_b:,}  mean={a.mean_b:.4f}  sd={a.sd_b:.4f}".replace(",", " "))
    print(f"Разница: {diff:+.4f} ({fmt_pct(rel, 1)} отн.)")
    print(f"{int((1 - a.alpha) * 100)}% ДИ: [{ci[0]:+.4f} ; {ci[1]:+.4f}]")
    print(f"z = {z:.3f}   p-value = {p_value:.4f} → "
          f"{'значимо' if p_value < a.alpha else 'НЕ значимо'} при alpha={a.alpha}")
    print(RULE)
    cv = max(a.sd_a / a.mean_a if a.mean_a else 0, a.sd_b / a.mean_b if a.mean_b else 0)
    if cv > 3:
        print("⚠ Метрика очень шумная (CV > 3) — типично для ARPU с выбросами. Нормальное приближение")
        print("  здесь хрупкое: проверь результат на винзоризованных данных или бутстрапом.")
    print("Проверки перед решением — те же 6 пунктов, что и для конверсий (см. `eval`).")


# ---------------------------------------------------------------- SRM и множественные сравнения

def srm_pvalue(n_a, n_b, split=0.5):
    total = n_a + n_b
    exp_a, exp_b = total * split, total * (1 - split)
    chi2 = (n_a - exp_a) ** 2 / exp_a + (n_b - exp_b) ** 2 / exp_b
    return two_sided_p(math.sqrt(chi2))  # 1 ст. свободы: p = 2·(1−Φ(√χ²))


def cmd_srm(a):
    p = srm_pvalue(a.n_a, a.n_b, a.split)
    total = a.n_a + a.n_b
    print(RULE)
    print("SRM — проверка соотношения групп")
    print(RULE)
    print(f"Факт:    A={a.n_a:,}  B={a.n_b:,}  (доля A = {a.n_a / total:.4f})".replace(",", " "))
    print(f"Ожидание: доля A = {a.split:.4f}")
    print(f"chi2 p-value = {p:.6f}")
    if p < 0.001:
        print("\n⛔ SRM ОБНАРУЖЕН. Рандомизация или логирование сломаны.")
        print("   Результат теста читать нельзя ни при каких p-value целевой метрики.")
        print("   Проверь: фильтры-редиректы, падения одного варианта, ботов, кэш CDN,")
        print("   разное время экспозиции, потерю событий в одной из веток.")
    else:
        print("\n✅ Значимого перекоса нет (порог p < 0.001).")


def cmd_multi(a):
    ps = sorted(float(x) for x in a.pvalues.split(","))
    m = len(ps)
    bonf = a.alpha / m
    print(RULE)
    print("МНОЖЕСТВЕННЫЕ СРАВНЕНИЯ")
    print(RULE)
    print(f"Тестов/метрик: {m}, alpha={a.alpha}")
    print(f"Вероятность хотя бы одной ложной находки без поправки: "
          f"{1 - (1 - a.alpha) ** m:.1%}")
    print(f"\nBonferroni порог: {bonf:.5f}")
    print(f"{'p-value':>10} | {'Bonferroni':>11} | {'BH (FDR)':>9} | порог BH")
    passed_bh = 0
    for i, p in enumerate(ps, 1):
        if p <= i / m * a.alpha:
            passed_bh = i
    for i, p in enumerate(ps, 1):
        thr_bh = i / m * a.alpha
        print(f"{p:>10.4f} | {'прошёл' if p < bonf else 'отсеян':>11} | "
              f"{'прошёл' if i <= passed_bh else 'отсеян':>9} | {thr_bh:.5f}")
    print(RULE)
    print("Bonferroni — консервативен, годится для решения о раскатке.")
    print("BH (FDR) — мягче, годится для генерации гипотез.")
    print("Главное: целевая метрика должна быть выбрана ДО теста. Значимость, найденная перебором")
    print("метрик и сегментов постфактум, — это гипотеза для следующего теста, а не результат.")


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="Калькулятор A/B-тестов (growth-decision skill)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--alpha", type=float, default=0.05)
        p.add_argument("--power", type=float, default=0.8)
        p.add_argument("--sides", type=int, default=2, choices=[1, 2])

    s = sub.add_parser("size", help="размер выборки для конверсии")
    s.add_argument("--baseline", type=float, required=True, help="текущая конверсия, доля (0.04 = 4%%)")
    s.add_argument("--mde-abs", type=float, help="MDE в абсолютных долях (0.004 = +0.4 п.п.)")
    s.add_argument("--mde-rel", type=float, help="MDE относительный (0.10 = +10%% к baseline)")
    s.add_argument("--daily", type=float, help="подходящих пользователей в день (обе группы)")
    common(s)
    s.set_defaults(func=cmd_size)

    s = sub.add_parser("size-mean", help="размер выборки для средней метрики")
    s.add_argument("--mean", type=float, required=True)
    s.add_argument("--sd", type=float, required=True)
    s.add_argument("--mde-abs", type=float)
    s.add_argument("--mde-rel", type=float)
    s.add_argument("--daily", type=float)
    common(s)
    s.set_defaults(func=cmd_size_mean)

    s = sub.add_parser("mde", help="какой эффект поймаем при заданном трафике")
    s.add_argument("--baseline", type=float, required=True)
    s.add_argument("--n", type=int, help="n на группу")
    s.add_argument("--daily", type=float)
    s.add_argument("--days", type=float)
    common(s)
    s.set_defaults(func=cmd_mde)

    s = sub.add_parser("eval", help="оценка результата по конверсиям")
    s.add_argument("--n-a", type=int, required=True)
    s.add_argument("--c-a", type=int, required=True, help="конверсий в группе A")
    s.add_argument("--n-b", type=int, required=True)
    s.add_argument("--c-b", type=int, required=True)
    s.add_argument("--mde-abs", type=float)
    s.add_argument("--mde-rel", type=float)
    s.add_argument("--split", type=float, default=0.5, help="ожидаемая доля группы A")
    s.add_argument("--alpha", type=float, default=0.05)
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("eval-mean", help="оценка результата по средним")
    for name in ("n-a", "n-b"):
        s.add_argument(f"--{name}", type=int, required=True)
    for name in ("mean-a", "sd-a", "mean-b", "sd-b"):
        s.add_argument(f"--{name}", type=float, required=True)
    s.add_argument("--alpha", type=float, default=0.05)
    s.set_defaults(func=cmd_eval_mean)

    s = sub.add_parser("srm", help="проверка перекоса групп")
    s.add_argument("--n-a", type=int, required=True)
    s.add_argument("--n-b", type=int, required=True)
    s.add_argument("--split", type=float, default=0.5)
    s.set_defaults(func=cmd_srm)

    s = sub.add_parser("multi", help="поправка на множественные сравнения")
    s.add_argument("--pvalues", required=True, help="через запятую: 0.01,0.03,0.2")
    s.add_argument("--alpha", type=float, default=0.05)
    s.set_defaults(func=cmd_multi)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
