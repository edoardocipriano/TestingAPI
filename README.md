# Diabetes Prediction API

API per la predizione del diabete basata su FastAPI e PyTorch.

## Deployment su Vercel

Per effettuare il deployment su Vercel, seguire questi passaggi:

1. Installa la CLI di Vercel:
   ```
   npm install -g vercel
   ```

2. Accedi al tuo account Vercel:
   ```
   vercel login
   ```

3. Esegui il deployment:
   ```
   vercel
   ```

4. Segui le istruzioni a schermo per completare il deployment.

## Struttura del Progetto

- `app/`: Contiene il codice principale dell'applicazione
  - `main.py`: API FastAPI
  - `model.py`: Definizione del modello PyTorch
  - `schemas.py`: Schema dei dati di input/output
  - `diabetes_model.pth`: Modello addestrato
  - `column_transformer.pkl`: Transformer per il preprocessing

- `api/`: Endpoint per Vercel
  - `index.py`: Punto di ingresso per le Serverless Functions di Vercel

- `vercel.json`: Configurazione del deployment su Vercel 