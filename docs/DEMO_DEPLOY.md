# Evaluator demo deployment

Deploy the existing GitHub repository through Streamlit Community Cloud:

- Repository: `Lujain-Alahmari/Daleel`
- Branch: `main`
- Main file: `demo_app.py`
- Python: `3.11`

In Advanced settings, set secrets privately in TOML format:

```toml
GEMINI_API_KEY = "YOUR_NEW_PRIVATE_KEY"
DEMO_ACCESS_CODE = "CHOOSE_A_PRIVATE_EVALUATOR_CODE"
DEMO_MAX_REQUESTS = 20
```

Do not put the real key or access code in GitHub. Share the access code separately with evaluators.

This entry point uses the same DashboardRag retrieval, generator and evidence validation. It excludes policy classification and gap analysis. There is no classifier checkpoint requirement. E5 and BGE still download on first use and require substantial RAM; Community Cloud deployment may exceed available resources. No successful hosted run is claimed until it has been tested.

Requests are serialised and capped per server process, with a 30-second cooldown. The cap resets on restart and is not a durable usage or billing control. The underlying adapter stores local audit files and sends question/context to Gemini. Use non-sensitive evaluation questions.

After deployment test one legal question and one out-of-scope question, check sources and logs, then share the actual working `.streamlit.app` URL. If dependencies fail to install or memory is exhausted, keep the failure logs and use a host with sufficient memory or the local demonstration. Do not silently replace the evaluated retrieval route with a different algorithm.
