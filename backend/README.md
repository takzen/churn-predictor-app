# Churn Predictor API (Backend)

This directory contains the backend service for the Churn Predictor application. It's a robust REST API built with **FastAPI** that serves a pre-trained Scikit-learn model for predicting customer churn.

## Features

*   **Prediction Endpoint:** Exposes a `/predict` endpoint that takes customer data and returns a churn prediction in real-time.
*   **Data Validation:** Uses Pydantic for rigorous validation of incoming request data.
*   **Production-Ready:** Configured for deployment on cloud platforms like Render.

## Parent Repository

This backend is part of the main `churn-predictor-app`. The corresponding frontend can be found in the `/frontend` directory of the parent repository.