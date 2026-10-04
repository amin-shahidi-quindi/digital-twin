# Remaining working time of a work order

A web app around the thesis model: describe an order, or how far a running
order has got, and it returns the working hours still needed (p10, p50, p90)
with charts, next to what the MES plan says.

Four tabs:

1. **Order in progress** – enter the order and its progress so far.
2. **Plan a new order** – invent an order and a speed; see every checkpoint
   (notebook cell 35, as a page).
3. **Real orders (test period)** – replay real orders the model never saw,
   against what really happened.
4. **How it works** – the method and the test result (thesis Table 5.1).

## Files

| file | what it is |
|---|---|
| `app.py` | the web page |
| `predictor.py` | features, the three models, the prediction (same rules as the notebook) |
| `build_artifacts.py` | turns the raw CSV exports into `artifacts/`; run on your own PC |
| `artifacts/` | the ~700 KB of prepared data the app reads |
| `requirements.txt` | exact package versions, the same as the thesis environment |
| `assets/` | the Quindi logo and icon (from `ProRob.ProNetManager.Edge/wwwroot/images`) |
| `.streamlit/config.toml` | the Quindi colours and font, from the `QuindiTheme` of the Edge app |

The app never reads the 267 MB production file. It reads `artifacts/` and
trains the three models when the server starts (about 6 seconds).

## Run it on your computer

```
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501.

## Rebuild the artifacts (only when the data changes)

```
python build_artifacts.py --data ../Thesis_python/data_csv/
```

At the end it prints the overall result. With the thesis data it must show
680 rows, 11.6% -> 7.3%, and 73.5% of rows where the model is closer.

## Put it online for free (Streamlit Community Cloud)

1. Create a **private** repository on GitHub and push this folder to it,
   including `artifacts/`, `assets/` and `.streamlit/`.
2. Go to https://share.streamlit.io, sign in with GitHub, and choose
   **Create app** -> your repository -> branch `main` -> file `app.py`.
3. Under **Advanced settings**, choose Python **3.12**. Deploy.
4. In the app's **Settings -> Sharing**, make it private and add the email
   address of each person who should see it. They sign in with that email.

The first visit after a quiet period can take a minute: free apps sleep when
nobody uses them, and wake up on the next visit.

## Before you upload

`artifacts/` holds real production data from the plant: article codes,
order codes, machine names and production curves. Putting it on GitHub and
Streamlit's servers means a third party stores it. Check that the owner of
the data agrees before you push, and keep both the repository and the app
private.
