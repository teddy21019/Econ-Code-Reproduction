from __future__ import annotations
from statistics import mode
from mesa import DataCollector
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.model import CurrencySubstitutionModel

def report_ex_rate(model:CurrencySubstitutionModel):
    c1,c2 = model.currency_prices
    return c1/c2

def report_inf_rate(model: CurrencySubstitutionModel):
    return model._p1 / model._Lp1

model_reporter = {
    "Exchange Rate" : report_ex_rate,
    "Inflation Rate": report_inf_rate
}

agent_reporter = {
    "First Consumption" : 'consumption_1'
}

abm_observer = model_reporter