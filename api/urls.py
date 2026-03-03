from django.urls import path
from . import views

urlpatterns = [
    # POST /api/analyze/  — Run the full LangGraph AI pipeline
    path("analyze/", views.AnalyzeView.as_view(), name="analyze"),

    # GET  /api/history/  — Fetch decision ledger as JSON
    path("history/", views.HistoryView.as_view(), name="history"),

    # GET  /api/trust-scores/  — Fetch current trust multipliers
    path("trust-scores/", views.TrustScoresView.as_view(), name="trust-scores"),

    # POST /api/evaluate/  — Trigger T+5 hindsight evaluation
    path("evaluate/", views.EvaluateView.as_view(), name="evaluate"),
]
