# app/engine.py
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Request, Form
from fastapi.responses import FileResponse, Response, JSONResponse
from .security import get_current_user, create_access_token, verify_password
from .db import get_hana_connection
from app.models import LoginRequest, ForecastInput, UploadCleanedData, DeleteFileRequest
import os
import csv as csv_module
import json
import pandas as pd
import shutil
from urllib.parse import unquote
from typing import List, Dict, Optional
from io import StringIO
import numpy as np



router = APIRouter(prefix="/api/engine", tags=["engine"])
@router.get("/status")

def engine_status(
    current_user: str = Depends(get_current_user),
    conn = Depends(get_hana_connection)
):
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM USERS")
        count = cur.fetchone()[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HANA error: {e}")
    return {

        "engine": "MyAwesomeEngine",
        "status": "running",
        "hana_dummy_row_count": count,
        "user": current_user
    }

@router.post("/test")
def function1():
    return{
        "test": "success"
    }

UPLOAD_DIR = "uploaded_files"

@router.post("/upload-cleaned-data")
async def upload_cleaned_data(
    file: UploadFile = File(...),
    granularity: Optional[str] = Form("Overall"),
    timeBucket: Optional[str] = Form("Daily"),
    forecastHorizon: Optional[int] = Form(30),
    columnMappings: Optional[str] = Form("{}"),
    timeDependentVariables: Optional[str] = Form("[]"),
    current_user: str = Depends(get_current_user)
):
    try:
        filename = file.filename
        content = await file.read()

        try:
            csv_text = content.decode("utf-8")
            df = pd.read_csv(StringIO(csv_text))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

        # Validate required columns
        required_columns = ['Date', 'Demand']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_columns}")

        # Convert string form fields into usable Python objects
        try:
            column_mappings_dict = json.loads(columnMappings)
        except json.JSONDecodeError:
            column_mappings_dict = {}

        try:
            time_dependent_variables_list = json.loads(timeDependentVariables)
        except json.JSONDecodeError:
            time_dependent_variables_list = []

        try:
            df, summary = process_uploaded_data(
                df,
                granularity,
                timeBucket,
                forecastHorizon,
                column_mappings_dict
            )

            # Save processed file
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            file_path = os.path.join(UPLOAD_DIR, filename)
            df.to_csv(file_path, index=False)

            # Save metadata
            metadata_path = os.path.join(UPLOAD_DIR, os.path.splitext(filename)[0] + "_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(summary, f)

            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": f"File '{filename}' processed successfully",
                    "filename": filename,
                    "summary": summary
                }
            )

        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Error processing data: {str(e)}",
                    "traceback": traceback_str
                }
            )

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Unexpected error: {str(e)}"}
        )

# Function to determine item_col based on forecast type
def determine_item_col(df, forecast_type):
    """Determines the appropriate item_col based on forecast type and available columns"""
    if forecast_type == "Item-wise":
        if "ProductID" in df.columns:
            return "ProductID"
    elif forecast_type == "Store-Item Combination":
        if "StoreID" in df.columns and "ProductID" in df.columns:
            # Create store_item column if it doesn't exist
            if "store_item" not in df.columns:
                df["store_item"] = df["StoreID"].astype(str) + " - " + df["ProductID"].astype(str)
            return "store_item"
    return None

    
# Update the data preprocessing section in the /upload-cleaned-data endpoint handler
def process_uploaded_data(df, granularity, time_bucket, forecast_horizon, column_mappings):
    try:
        # Clean up granularity and time bucket values
        granularity = granularity.strip() if isinstance(granularity, str) else granularity
        time_bucket = time_bucket.strip() if isinstance(time_bucket, str) else time_bucket
        
        # Map granularity to forecast_type
        forecast_type = map_granularity_to_forecast_type(granularity)
        
        # Determine item_col based on forecast_type
        item_col = determine_item_col(df, forecast_type)
        
        # Create summary
        stats = {
            "mean": float(df["Demand"].mean()),
            "median": float(df["Demand"].median()),
            "min": float(df["Demand"].min()),
            "max": float(df["Demand"].max()),
            "std": float(df["Demand"].std())
        }
        
        summary = {
            "originalColumns": df.columns.tolist(),
            "rowCount": len(df),
            "granularity": granularity,
            "timeBucket": time_bucket,
            "forecastHorizon": forecast_horizon,
            "item_col": item_col,
            "dataFrequency": api_infer_frequency(df),
            "seasonality": api_detect_seasonality(df),
            "dataStats": stats,
            "processedColumns": df.columns.tolist(),
            "forecast_type": forecast_type
        }
        
        return df, summary
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise ValueError(f"Data preprocessing failed: {str(e)}")
    
# Simplified version of detect_seasonality for API use
def api_detect_seasonality(df):
    """Detect seasonality patterns without Streamlit dependencies"""
    try:
        if len(df) < 4:
            print(f"Dataset too small ({len(df)} rows) for seasonality detection. Defaulting to period=7.")
            return 7

        if 'Date' not in df.columns or 'Demand' not in df.columns:
            print("Required columns 'Date' and 'Demand' must be present")
            return 7
            
        # Infer frequency from date index
        date_freq = api_infer_frequency(df)
        
        freq_based_period = {
            'D': 7,    # Daily -> Weekly seasonality
            'W': 52,   # Weekly -> Yearly seasonality
            'M': 12,   # Monthly -> Yearly seasonality
            'Q': 4,    # Quarterly -> Yearly seasonality
            'Y': 1     # Yearly -> No seasonality
        }.get(date_freq, 7)  # Default to weekly if unknown
        
        return freq_based_period
        
    except Exception as e:
        print(f"Error detecting seasonality: {str(e)}")
        return 7  # Default to weekly seasonality

# Simplified version of infer_frequency for API use
def api_infer_frequency(df, date_col='Date'):
    """Infer the frequency of time series data without Streamlit dependencies"""
    try:
        if date_col not in df.columns:
            print(f"'{date_col}' not found in DataFrame columns")
            return 'D'  # Default to daily
        
        # Get the datetime series
        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
        
        if len(dates) < 2:
            print("Too few dates to infer frequency. Defaulting to 'D'.")
            return 'D'
        
        # Calculate differences between consecutive dates
        date_diffs = dates.sort_values().diff().dt.days.dropna()
        
        if date_diffs.empty:
            print("No valid differences to infer frequency. Defaulting to 'D'.")
            return 'D'
        
        # Get the most common difference
        most_common_diff = date_diffs.value_counts().index[0]
        
        # Map to frequency based on most common difference
        freq_map = {
            1: 'D',     # Daily
            7: 'W',     # Weekly
            14: 'W',    # Bi-weekly
            28: 'M',    # Monthly (28 days)
            30: 'M',    # Monthly (30 days)
            31: 'M',    # Monthly (31 days)
            90: 'Q',    # Quarterly (90 days)
            91: 'Q',    # Quarterly (91 days)
            92: 'Q',    # Quarterly (92 days)
            365: 'Y',   # Yearly (365 days)
            366: 'Y'    # Yearly (366 days - leap year)
        }
        
        # Get frequency or default to daily
        freq = freq_map.get(most_common_diff, 'D')
        
        print(f"Inferred frequency: {freq} (most common diff: {most_common_diff} days)")
        return freq
        
    except Exception as e:
        print(f"Error inferring frequency: {str(e)}")
        return 'D'  # Default to daily on failure

@router.get("/files")
async def list_uploaded_files(current_user: str = Depends(get_current_user)):
    try:
        if not os.path.exists(UPLOAD_DIR):
            return {"status": "success", "files": []}
        
        files = os.listdir(UPLOAD_DIR)
        return {"status": "success", "files": files}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving file list: {str(e)}")

@router.get("/models")
async def get_all_models(current_user: str = Depends(get_current_user), conn = Depends(get_hana_connection)):
    try:       
        with conn.cursor() as cur:
            cur.execute("SELECT MODELNAME FROM DBADMIN.MODELS")
            models = [row[0] for row in cur.fetchall()]
        return {"models":models}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/run-forecast")
async def run_forecast(request: Request, current_user: str = Depends(get_current_user), conn = Depends(get_hana_connection)):
    try:
        data = await request.json()
        filename: Optional[str] = data.get("filename")
        granularity: str = data.get("granularity", "Overall")
        forecast_horizon: int = int(data.get("forecastHorizon", 30))
        time_bucket: str = data.get("timeBucket", "Daily")
        forecast_lock: int = int(data.get("forecastLock", 0))
        selected_models: List[str] = data.get("selectedModels", ["ARIMA", "SARIMA", "Prophet"])
        time_dependent_variables: List[str] = data.get("timeDependentVariables", [])
        column_mappings: Dict = data.get("columnMappings", {})

        if not filename:
            raise HTTPException(status_code=400, detail="Filename required")

        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        try:
            result = run_forecast_for_file(
                conn,
                file_path,
                granularity,
                forecast_horizon,
                selected_models,
                time_dependent_variables,
                time_bucket,
                forecast_lock,
                column_mappings
            )

            serializable_result = make_json_serializable(result)
            try:
                json.dumps(serializable_result)  # test if serializable
                return JSONResponse(content=serializable_result, status_code=200)
            except TypeError as te:
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": f"Error serializing results: {str(te)}"}
                )

        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            return JSONResponse(
                status_code=200,
                content={
                    "status": "error",
                    "message": f"Error running forecast: {str(e)}",
                    "traceback": traceback_str
                }
            )

    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Invalid JSON in request: {str(e)}"}
        )
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Server error: {str(e)}", "traceback": traceback_str}
        )

# Function to map frontend granularity to backend forecast_type
def map_granularity_to_forecast_type(granularity):
    """Maps frontend granularity options to backend forecast_type values"""
    granularity_map = {
        "Overall": "Overall",
        "ProductID-wise": "Item-wise",
        "StoreID-ProductID Combination": "Store-Item Combination",
        # Add variations to handle potential truncation or misspellings
        "StoreID-ProductID Combinat": "Store-Item Combination",
        "StoreID-ProductID": "Store-Item Combination"
    }
    # Clean up the input granularity
    cleaned_granularity = granularity.strip() if isinstance(granularity, str) else granularity
    return granularity_map.get(cleaned_granularity, "Overall")

# Function to ensure dates are properly formatted
def ensure_date_format(df):
    """Ensure dates are in the correct format and properly set as datetime"""
    try:
        # Check if Date is already datetime
        if pd.api.types.is_datetime64_dtype(df['Date']):
            print("Date column is already in datetime format")
            return df
            
        # Convert to datetime with flexible parsing
        original_dates = df['Date'].copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
        if df['Date'].isna().any():
            # Try specific formats
            formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y']
            
            for fmt in formats:
                try:
                    test_dates = pd.to_datetime(original_dates, format=fmt, errors='coerce')
                    if test_dates.isna().sum() < df['Date'].isna().sum():
                        df['Date'] = test_dates
                        print(f"Converted dates using format {fmt}")
                        break
                except:
                    continue
            
        # Check if we still have NaT values and filter them out if necessary
        nat_count = df['Date'].isna().sum()
        if nat_count > 0:
            print(f"Warning: {nat_count} dates could not be parsed and will be removed")
            df = df.dropna(subset=['Date'])
            
        # Sort by date
        df = df.sort_values(by='Date')
        
        return df
    except Exception as e:
        print(f"Error in date formatting: {str(e)}")
        return df  # Return original DataFrame if we can't fix it
    
# Function to ensure required ID columns exist based on granularity
def ensure_id_columns_exist(df, forecast_type):
    """Ensure required ID columns exist based on the forecast type"""
    if forecast_type == "Item-wise" and "ProductID" not in df.columns:
        print("ProductID column missing, creating a default")
        df["ProductID"] = "default_product"
    
    if forecast_type == "Store-Item Combination":
        # Create default columns for StoreID and ProductID if needed
        if "StoreID" not in df.columns:
            print("StoreID column missing, creating default")
            df["StoreID"] = "default_store"
        if "ProductID" not in df.columns:
            print("ProductID column missing, creating default")
            df["ProductID"] = "default_product"
            
        # Create or update store_item column with consistent separator
        print("Creating/updating store_item column")
        df["store_item"] = df["StoreID"].astype(str).str.strip() + " - " + df["ProductID"].astype(str).str.strip()
        
        # Validate the store_item column
        if df["store_item"].isnull().any():
            print("Warning: Some store_item combinations contain null values")
            df["store_item"] = df["store_item"].fillna("unknown - unknown")
    
    return df

# Function to safely call forecast_models with proper date handling
def safe_forecast_models(df, selected_models, additional_cols=None, item_col=None, forecast_type='Overall', horizon=30):
    """A wrapper for forecast_models that ensures proper date handling"""
    try:
        # Check if Date is the index
        if isinstance(df.index, pd.DatetimeIndex) and 'Date' not in df.columns:
            # Reset index to get Date as a column
            df = df.reset_index()
            print("Reset DatetimeIndex to column 'Date'")
        
        # Make sure Date column exists and is not already an index
        if 'Date' in df.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])
            
            # Now we're ready to call the original function
            from app import forecast_models
            return forecast_models(df, selected_models, additional_cols, item_col, forecast_type, horizon)
        else:
            raise ValueError("DataFrame must have a 'Date' column or DatetimeIndex")
    
    except Exception as e:
        print(f"Error in safe_forecast_models: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
    
# Function to detect and adapt forecast data structure based on model
def adapt_forecast_structure(future_forecasts, forecast_type, selected_models=None, results=None):
    """
    Adapts the forecast data structure to a standard format based on what's detected.
    Handles special cases like Prophet model results.
    
    Parameters:
    -----------
    future_forecasts : dict or list
        The future forecasts from the forecast_models function
    forecast_type : str
        The type of forecast: "Overall", "Item-wise", or "Store-Item Combination"
    selected_models : list, optional
        The models that were selected for forecasting
    results : dict, optional
        The results dictionary from forecast_models function
        
    Returns:
    --------
    dict
        A standardized version of the future_forecasts dictionary
    """
    # If it's already empty or None, return a default structure
    if not future_forecasts:
        return {"Default": [0.0] * 30}  # Return a dummy forecast
    
    # Helper function to determine the most appropriate model name
    def get_model_name():
        if selected_models and len(selected_models) == 1:
            return selected_models[0]  # If only one model was selected, use that
        elif isinstance(results, dict) and "model_name" in results:
            return results["model_name"]
        elif selected_models and "Prophet" in selected_models:
            return "Prophet"  # Prefer Prophet if it was one of the selected models
        elif selected_models and len(selected_models) > 0:
            return selected_models[0]  # Use the first selected model
        else:
            return "Default"
            
    # If the top level is not a dict, wrap it
    if not isinstance(future_forecasts, dict):
        print(f"future_forecasts is not a dict, it's a {type(future_forecasts)}")
        # If it's a list, create a default model but try to infer the model name
        if isinstance(future_forecasts, list):
            model_name = get_model_name()
            return {model_name: future_forecasts}
        # If it's something else, return a default structure
        return {get_model_name(): [0.0] * 30}  # Return a dummy forecast
    
    # Create a copy to avoid modifying the original
    adapted_forecasts = {}
    
    # Handle Prophet special case for Overall forecasts (single level)
    if forecast_type == "Overall":
        for model_name, forecasts in future_forecasts.items():
            if isinstance(forecasts, dict):
                # For nested dictionaries (like Prophet might produce)
                adapted_forecasts[model_name] = {}
                for sub_key, sub_forecasts in forecasts.items():
                    # Ensure sub_forecasts is a list
                    if not isinstance(sub_forecasts, list):
                        sub_forecasts = [float(sub_forecasts)]
                    adapted_forecasts[model_name][sub_key] = sub_forecasts
            else:
                # For direct list forecasts
                # Ensure forecasts is a list
                if not isinstance(forecasts, list):
                    forecasts = [float(forecasts)]
                adapted_forecasts[model_name] = forecasts
                
    # Handle Prophet special case for Item-wise forecasts (two levels)
    elif forecast_type == "Item-wise":
        for item_id, item_forecasts in future_forecasts.items():
            adapted_forecasts[item_id] = {}
            # Check if item_forecasts is already a dictionary
            if isinstance(item_forecasts, dict):
                for model_name, forecasts in item_forecasts.items():
                    # Ensure forecasts is a list
                    if not isinstance(forecasts, list):
                        forecasts = [float(forecasts)]
                    adapted_forecasts[item_id][model_name] = forecasts
            else:
                # If it's not a dict, ensure it's a list
                if not isinstance(item_forecasts, list):
                    item_forecasts = [float(item_forecasts)]
                # If it's a list, create a "Default" model for it
                adapted_forecasts[item_id]["Default"] = item_forecasts
            
    # Handle Prophet special case for Store-Item Combination forecasts (two levels)
    elif forecast_type == "Store-Item Combination":
        for combo_id, combo_forecasts in future_forecasts.items():
            adapted_forecasts[combo_id] = {}
            # Check if combo_forecasts is already a dictionary
            if isinstance(combo_forecasts, dict):
                for model_name, forecasts in combo_forecasts.items():
                    # Ensure forecasts is a list
                    if not isinstance(forecasts, list):
                        forecasts = [float(forecasts)]
                    adapted_forecasts[combo_id][model_name] = forecasts
            else:
                # If it's not a dict, ensure it's a list
                if not isinstance(combo_forecasts, list):
                    combo_forecasts = [float(combo_forecasts)]
                # If it's a list, create a "Default" model for it
                adapted_forecasts[combo_id]["Default"] = combo_forecasts
                
    return adapted_forecasts

# Function to transform forecast results into a CSV-ready format
def transform_forecasts_for_csv(results, future_forecasts, dates, forecast_type, future_dates=None):
    """Transform forecast results into a CSV-ready format"""
    csv_rows = []
    
    # Generate future dates if not provided
    if future_dates is None and isinstance(dates, list) and len(dates) > 0:
        try:
            last_date = pd.to_datetime(dates[-1])
            forecast_length = 30  # Default
            if isinstance(future_forecasts, dict) and len(future_forecasts) > 0:
                first_item = next(iter(future_forecasts.values()))
                if isinstance(first_item, dict) and len(first_item) > 0:
                    first_model = next(iter(first_item.values()))
                    if isinstance(first_model, list):
                        forecast_length = len(first_model)
            
            future_dates = [
                (last_date + pd.DateOffset(days=i+1)).strftime('%Y-%m-%d')
                for i in range(forecast_length)
            ]
        except Exception as e:
            print(f"Error generating future dates: {e}")
            future_dates = [f"Future_{i+1}" for i in range(30)]  # Fallback
    
    # Case 1: Overall forecast
    if forecast_type == "Overall":
        for model_name, forecasts in future_forecasts.items():
            if isinstance(forecasts, dict) and hasattr(forecasts, 'items'):
                for inner_key, forecast_values in forecasts.items():
                    for i, forecast_value in enumerate(forecast_values):
                        if i < len(future_dates):
                            row = {
                                "Date": future_dates[i],
                                "Store": "",
                                "Item": inner_key,
                                "Model": model_name,
                                "Forecast": forecast_value
                            }
                            csv_rows.append(row)
            else:
                for i, forecast_value in enumerate(forecasts):
                    if i < len(future_dates):
                        row = {
                            "Date": future_dates[i],
                            "Store": "",
                            "Item": "Total",
                            "Model": model_name,
                            "Forecast": forecast_value
                        }
                        csv_rows.append(row)
    
    # Case 2: Item-wise forecast
    elif forecast_type == "Item-wise":
        for item_id, item_forecasts in future_forecasts.items():
            if isinstance(item_forecasts, dict) and hasattr(item_forecasts, 'items'):
                for model_name, forecasts in item_forecasts.items():
                    for i, forecast_value in enumerate(forecasts):
                        if i < len(future_dates):
                            item_name = str(item_id).strip() if item_id and str(item_id).strip() else "Unknown Item"
                            row = {
                                "Date": future_dates[i],
                                "Store": "",
                                "Item": item_name,
                                "Model": model_name,
                                "Forecast": forecast_value
                            }
                            csv_rows.append(row)
            else:
                for i, forecast_value in enumerate(item_forecasts):
                    if i < len(future_dates):
                        item_name = str(item_id).strip() if item_id and str(item_id).strip() else "Unknown Item"
                        row = {
                            "Date": future_dates[i],
                            "Store": "",
                            "Item": item_name,
                            "Model": "Default",
                            "Forecast": forecast_value
                        }
                        csv_rows.append(row)
    
    # Case 3: Store-Item Combination forecast
    elif forecast_type == "Store-Item Combination":
        for combo_id, combo_forecasts in future_forecasts.items():
            # Split the combo ID using the consistent separator
            store_id = "Unknown Store"
            item_id = "Unknown Item"
            if isinstance(combo_id, str):
                parts = combo_id.split(" - ", 1)
                if len(parts) == 2:
                    store_id, item_id = parts
                    store_id = store_id.strip()
                    item_id = item_id.strip()
            
            if isinstance(combo_forecasts, dict) and hasattr(combo_forecasts, 'items'):
                for model_name, forecasts in combo_forecasts.items():
                    for i, forecast_value in enumerate(forecasts):
                        if i < len(future_dates):
                            row = {
                                "Date": future_dates[i],
                                "Store": store_id,
                                "Item": item_id,
                                "Model": model_name,
                                "Forecast": forecast_value
                            }
                            csv_rows.append(row)
            else:
                for i, forecast_value in enumerate(combo_forecasts):
                    if i < len(future_dates):
                        row = {
                            "Date": future_dates[i],
                            "Store": store_id,
                            "Item": item_id,
                            "Model": "Default",
                            "Forecast": forecast_value
                        }
                        csv_rows.append(row)
    
    return csv_rows
    
# Dummy placeholders for your actual logic
def run_forecast_for_file(conn, file_path, granularity, forecast_horizon, selected_models, time_dependent_variables, time_bucket, forecast_lock, column_mappings):
    """Run forecasting on a file using the specified parameters"""
    try:
    
        with conn.cursor() as cur:
            cur.execute("SELECT MODELNAME FROM DBADMIN.MODELS")
            models = [row[0] for row in cur.fetchall()]
        # return {"models":models}
        available_model_namess = models
    
        # Set default models if none provided
        if selected_models is None or not selected_models:
            selected_models = ["ARIMA", "SARIMA", "Prophet"]
        
        # Validate selected models against available models
        available_model_names = [model for model in available_model_namess]
        invalid_models = [model for model in selected_models if model not in available_model_names]
        if invalid_models:
            raise ValueError(f"Invalid model(s) selected: {invalid_models}. Available models are: {available_model_names}")
        
        # Filter out any invalid models
        selected_models = [model for model in selected_models if model in available_model_names]
        if not selected_models:
            raise ValueError("No valid models selected for forecasting")
        
        print(f"Using validated models: {selected_models}")
        
        # Check if metadata file exists for this file
        metadata_path = os.path.join(UPLOAD_DIR, os.path.splitext(os.path.basename(file_path))[0] + "_metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Loaded metadata from {metadata_path}")
                
                # If granularity was overridden in the request, note that
                if granularity != metadata.get("granularity"):
                    print(f"Granularity changed from {metadata.get('granularity')} to {granularity}")
                    
                # Update forecast_type if it's in the metadata
                stored_forecast_type = metadata.get("forecast_type")
                if stored_forecast_type:
                    forecast_type_from_granularity = map_granularity_to_forecast_type(granularity)
                    if stored_forecast_type != forecast_type_from_granularity:
                        print(f"Warning: Stored forecast_type ({stored_forecast_type}) differs from current granularity mapping ({forecast_type_from_granularity})")
                        print(f"Using new forecast_type: {forecast_type_from_granularity}")
                        
                # Get item_col from metadata if available
                item_col_from_metadata = metadata.get("item_col")
                if item_col_from_metadata:
                    print(f"Using item_col from metadata: {item_col_from_metadata}")
            except Exception as e:
                print(f"Error loading metadata: {e}")
        
        # Read the CSV file
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Loaded DataFrame with columns: {df.columns.tolist()}")
        print(f"Using column mappings: {column_mappings}")
        
        # Get the mapped columns from column_mappings
        date_col = column_mappings.get("Date")
        demand_col = column_mappings.get("Demand")
        product_id_col = column_mappings.get("ProductID")
        store_id_col = column_mappings.get("StoreID")
        
        print(f"Looking for mapped Date column: {date_col}")
        print(f"Looking for mapped Demand column: {demand_col}")
        print(f"Looking for mapped ProductID column: {product_id_col}")
        print(f"Looking for mapped StoreID column: {store_id_col}")
        
        # Ensure all required columns exist and are named correctly
        # First, handle the Demand column
        if demand_col and demand_col in df.columns:
            print(f"Renaming '{demand_col}' to 'Demand'")
            # If Demand column already exists and is different from the mapped column, drop it
            if 'Demand' in df.columns and demand_col != 'Demand':
                print(f"Dropping existing 'Demand' column in favor of mapped column '{demand_col}'")
                df.drop('Demand', axis=1, inplace=True)
            df.rename(columns={demand_col: 'Demand'}, inplace=True)
        
        # Next, handle the Date column
        if date_col and date_col in df.columns:
            print(f"Renaming '{date_col}' to 'Date'")
            # If Date column already exists and is different from the mapped column, drop it
            if 'Date' in df.columns and date_col != 'Date':
                print(f"Dropping existing 'Date' column in favor of mapped column '{date_col}'")
                df.drop('Date', axis=1, inplace=True)
            df.rename(columns={date_col: 'Date'}, inplace=True)
        
        # Handle the ProductID column
        if product_id_col and product_id_col in df.columns:
            print(f"Renaming '{product_id_col}' to 'ProductID'")
            # If ProductID column already exists and is different from the mapped column, drop it
            if 'ProductID' in df.columns and product_id_col != 'ProductID':
                print(f"Dropping existing 'ProductID' column in favor of mapped column '{product_id_col}'")
                df.drop('ProductID', axis=1, inplace=True)
            df.rename(columns={product_id_col: 'ProductID'}, inplace=True)
        
        # Handle the StoreID column
        if store_id_col and store_id_col in df.columns:
            print(f"Renaming '{store_id_col}' to 'StoreID'")
            # If StoreID column already exists and is different from the mapped column, drop it
            if 'StoreID' in df.columns and store_id_col != 'StoreID':
                print(f"Dropping existing 'StoreID' column in favor of mapped column '{store_id_col}'")
                df.drop('StoreID', axis=1, inplace=True)
            df.rename(columns={store_id_col: 'StoreID'}, inplace=True)
        
        # Ensure Date column exists and is properly formatted
        if 'Date' not in df.columns:
            print("Date column missing, checking for mapped column")
            
            if date_col:
                print(f"Mapped Date column '{date_col}' was not found in the DataFrame")
                
            # Look for alternative date columns
            possible_date_cols = [col for col in df.columns if col.lower().find('date') >= 0 or 
                                 col.lower().find('time') >= 0 or 
                                 col.lower().find('day') >= 0 or 
                                 col.lower().find('month') >= 0 or
                                 col.lower().find('year') >= 0]
            
            if possible_date_cols:
                print(f"Found potential date columns: {possible_date_cols}")
                # Try to identify the most likely date column
                for col in possible_date_cols:
                    # Convert to string first to handle numeric date formats
                    test_series = df[col].astype(str)
                    # Try to convert to datetime
                    test_dates = pd.to_datetime(test_series, errors='coerce')
                    # If at least 90% of values are valid dates, use this column
                    if test_dates.notna().mean() >= 0.9:
                        print(f"Using '{col}' as the Date column (>90% valid dates)")
                        df.rename(columns={col: 'Date'}, inplace=True)
                        break
                else:
                    # If no column has >90% valid dates, use the first column in the list
                    print(f"Using '{possible_date_cols[0]}' as the Date column (best guess)")
                    df.rename(columns={possible_date_cols[0]: 'Date'}, inplace=True)
            else:
                raise ValueError("No Date column found in the DataFrame. Please map a date column to 'Date'.")
        
        print(f"Using column '{df.columns[df.columns == 'Date'][0]}' as the Date column")
        
        # Try to convert Date column to datetime with multiple formats
        date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y',
                        '%Y%m%d', '%d%m%Y', '%m%d%Y', '%b %d, %Y', '%B %d, %Y',
                        '%d %b %Y', '%d %B %Y', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S']
        
        # First try automatic conversion
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # If automatic conversion fails, try specific formats
        if df['Date'].isna().any():
            original_series = df['Date'].copy()
            for date_format in date_formats:
                try:
                    df['Date'] = pd.to_datetime(original_series, format=date_format, errors='coerce')
                    # If this format works for most values, use it
                    if df['Date'].notna().mean() > 0.9:
                        print(f"Successfully converted dates using format: {date_format}")
                        break
                except Exception:
                    continue
        
        # Check for NaT values after conversion
        if df['Date'].isna().any():
            print(f"Warning: Date column contains {df['Date'].isna().sum()} NaT values after conversion")
            # Remove rows with invalid dates
            orig_len = len(df)
            df = df.dropna(subset=['Date'])
            print(f"Removed {orig_len - len(df)} rows with invalid dates")
            
            if len(df) == 0:
                raise ValueError("No valid dates found after cleaning. Please check your date format.")
        
        # Sort by date
        df = df.sort_values(by='Date')
        
        # Ensure proper date formatting
        df = ensure_date_format(df)
        
        # Ensure Demand column exists
        if 'Demand' not in df.columns:
            print("Demand column missing, checking for mapped column")
            
            if demand_col:
                print(f"Mapped Demand column '{demand_col}' was not found in the DataFrame")
                
            # Look for potential numeric columns that might contain demand data
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                print(f"Found potential numeric columns for Demand: {numeric_cols}")
                # Rename the first numeric column to 'Demand'
                first_numeric = numeric_cols[0]
                print(f"Using '{first_numeric}' as the Demand column")
                df.rename(columns={first_numeric: 'Demand'}, inplace=True)
            else:
                raise ValueError("No suitable Demand column found in the DataFrame. Please map a numeric column to 'Demand'.")
                
        print(f"Using column '{df.columns[df.columns == 'Demand'][0]}' as the Demand column")
        
        # Convert Demand column to numeric
        df['Demand'] = pd.to_numeric(df['Demand'], errors='coerce')
        if df['Demand'].isna().sum() > 0:
            print(f"Warning: Demand column contains {df['Demand'].isna().sum()} NaN values after conversion")
            print("Applying imputation to fill missing values...")
            
            # Try interpolation first
            df_temp = df.copy()
            df_temp['Demand'] = df_temp['Demand'].interpolate(method='linear')
            
            # If there are still NaNs, use forward fill
            if df_temp['Demand'].isna().any():
                df_temp['Demand'] = df_temp['Demand'].fillna(method='ffill')
            
            # If there are still NaNs, use backward fill
            if df_temp['Demand'].isna().any():
                df_temp['Demand'] = df_temp['Demand'].fillna(method='bfill')
            
            # If there are still NaNs, fill with 0 as last resort
            if df_temp['Demand'].isna().any():
                df_temp['Demand'] = df_temp['Demand'].fillna(0)
                
            df = df_temp
            print(f"After imputation, NaN count: {df['Demand'].isna().sum()}")
        
        # Check for negative values in Demand
        neg_count = (df['Demand'] < 0).sum()
        if neg_count > 0:
            print(f"Warning: Demand column contains {neg_count} negative values")
            # For forecasting, we might want to either:
            # 1. Replace negatives with 0 (for counts/sales that can't be negative)
            # 2. Keep as is (for inventory changes, temperature, etc.)
        
        # Convert the granularity to forecast_type
        forecast_type = map_granularity_to_forecast_type(granularity)
        
        # Ensure required ID columns exist based on forecast type
        df = ensure_id_columns_exist(df, forecast_type)
        
        print(f"Running forecast with type: {forecast_type}, horizon: {forecast_horizon}, time bucket: {time_bucket}, forecast lock: {forecast_lock}")
        print(f"DataFrame shape: {df.shape}, date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Import the forecast_models function from app.py
        from app import forecast_models
        
        # Determine the appropriate item_col based on forecast_type
        item_col = None
        # First check if we have a valid item_col from metadata
        if 'item_col_from_metadata' in locals() and item_col_from_metadata and item_col_from_metadata in df.columns:
            print(f"Using item_col from metadata: {item_col_from_metadata}")
            item_col = item_col_from_metadata
        # Otherwise, determine based on forecast_type
        elif forecast_type == "Item-wise":
            if "ProductID" in df.columns:
                item_col = "ProductID"
                print(f"Using ProductID as item_col for Item-wise forecast")
            else:
                print(f"ProductID column not found, creating default")
                df["ProductID"] = "default_product"
                item_col = "ProductID"
        elif forecast_type == "Store-Item Combination":
            # Create a combined column for store-item if needed
            if "store_item" in df.columns:
                item_col = "store_item"
                print(f"Using existing store_item column")
            else:
                if "StoreID" in df.columns and "ProductID" in df.columns:
                    print(f"Creating store_item column from StoreID and ProductID")
                    df["store_item"] = df["StoreID"].astype(str) + "_" + df["ProductID"].astype(str)
                else:
                    print(f"StoreID or ProductID columns missing, creating default store_item")
                    df["store_item"] = "default_store_item"
                item_col = "store_item"
        
        # Important safety measures to prevent statsmodels errors
        # Make sure the Date column is the index for time series functions
        if 'Date' in df.columns:
            # Convert to datetime again just to be super safe
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])  # Remove rows with invalid dates
            
            # Set Date as index for time series models
            df_indexed = df.set_index('Date').sort_index()
            
            # Check if we have enough data points
            if len(df_indexed) < 10:
                raise ValueError(f"Not enough valid data points after cleaning. Only {len(df_indexed)} rows remaining.")
            
            # Make a copy to prevent modification warnings
            df_for_forecast = df_indexed.copy()
        else:
            raise ValueError("Date column is missing from the DataFrame after preprocessing.")
        
        # Run the forecast using the indexed DataFrame
        print(f"Running forecast models with DataFrame shape: {df_for_forecast.shape}")
        
        # Use our safe wrapper function to ensure proper date handling
        results, future_forecasts, dates = safe_forecast_models(
            df_for_forecast, 
            selected_models,
            additional_cols=time_dependent_variables,
            item_col=item_col,
            forecast_type=forecast_type,
            horizon=forecast_horizon
        )
        
        # Convert dates to serializable format if needed
        if isinstance(dates, pd.Series) or isinstance(dates, pd.DatetimeIndex):
            dates_list = dates.tolist()
        else:
            dates_list = dates
            
        # Make all results JSON-serializable
        response_data = {
            "status": "success",
            "results": make_json_serializable(results),
            "future_forecasts": make_json_serializable(future_forecasts),
            "dates": make_json_serializable(dates_list),
            "forecast_type": forecast_type,  # Include the forecast type in the response
            "config": {
                "granularity": granularity,
                "forecast_horizon": forecast_horizon,
                "time_bucket": time_bucket,
                "forecast_lock": forecast_lock,
                "selected_models": selected_models,
                "column_mappings": column_mappings
            }
        }
        
        # For item-wise or store-item forecasts, add item identifiers to the CSV-ready format
        if forecast_type in ["Item-wise", "Store-Item Combination"]:
            # Get future dates (needed for CSV)
            if isinstance(dates_list, list) and dates_list:
                last_date = pd.to_datetime(dates_list[-1])
                future_dates = [
                    (last_date + pd.DateOffset(days=i+1)).strftime('%Y-%m-%d')
                    for i in range(forecast_horizon)
                ]
            else:
                future_dates = [f"Future_{i+1}" for i in range(forecast_horizon)]
                
            # Add CSV-ready data to the response with error handling
            try:
                # Print some debug info
                print(f"forecast_type: {forecast_type}")
                print(f"results data type: {type(results)}")
                print(f"future_forecasts data type: {type(future_forecasts)}")
                print(f"selected_models: {selected_models}")
                
                # Print the structure of future_forecasts in more detail
                print("future_forecasts structure:")
                if isinstance(future_forecasts, dict):
                    for key, value in future_forecasts.items():
                        print(f"  key: {key}, value type: {type(value)}")
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                print(f"    sub_key: {sub_key}, sub_value type: {type(sub_value)}")
                                if isinstance(sub_value, (list, tuple)) and len(sub_value) > 0:
                                    print(f"      first element type: {type(sub_value[0])}")
                        elif isinstance(value, (list, tuple)) and len(value) > 0:
                            print(f"    first element type: {type(value[0])}")
                
                if isinstance(future_forecasts, dict) and len(future_forecasts) > 0:
                    first_key = next(iter(future_forecasts))
                    print(f"first_key: {first_key}")
                    print(f"first value type: {type(future_forecasts[first_key])}")
                
                # Adapt the forecast structure to handle different formats
                adapted_forecasts = adapt_forecast_structure(future_forecasts, forecast_type, selected_models, results)
                
                csv_data = transform_forecasts_for_csv(
                    results, adapted_forecasts, dates_list, forecast_type, future_dates
                )
                response_data["csv_data"] = make_json_serializable(csv_data)
            except Exception as e:
                import traceback
                print(f"Error generating CSV data: {str(e)}")
                print(traceback.format_exc())
                # Continue without CSV data
                response_data["csv_error"] = str(e)
            
        return response_data
    except Exception as e:
        print(f"Error running forecast: {str(e)}")
        print(f"Error details: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Error running forecast: {str(e)}"
        }


def make_json_serializable(obj):
    """Convert pandas Timestamps, numpy types and other non-serializable objects to serializable types"""
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}  # Convert all keys to strings
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, pd.DatetimeIndex):
        return [d.strftime('%Y-%m-%d %H:%M:%S') for d in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, pd.Series):
        return make_json_serializable(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        try:
            # Try to convert to string if it's not a basic type
            return str(obj)
        except:
            return None
    
@router.get("/download/{file_name}")
async def download_file(file_name: str, current_user: str = Depends(get_current_user)):
    print(file_name)
    try:
        file_path = os.path.join(UPLOAD_DIR, file_name)

        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(path=file_path, filename=file_name, media_type='application/octet-stream')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

@router.get("/forecastdata")
async def get_forecasted_date(current_user: str = Depends(get_current_user), conn = Depends(get_hana_connection)):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM DBADMIN.FINAL_FORECASTS")
            result = cur.fetchall()

        forecasts = [dict(zip([column[0] for column in cur.description], row)) for row in result]
        return {"status": "success", "forecasts": forecasts}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving forecast data: {str(e)}") 

@router.post("/login")
async def login(request: LoginRequest, conn=Depends(get_hana_connection)):
    try:
        with conn.cursor() as cur:
            query = "SELECT PASSWORDHASH FROM DBADMIN.USERS WHERE username = ?"
            cur.execute(query, (request.username,))
            result = cur.fetchone()
            # print(result[0])
        if not result:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        hashed_password = result[0]
        if hashed_password != request.password:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        # if not verify_password(request.password, hashed_password):
        #     raise HTTPException(status_code=401, detail="Invalid username or password")

        # Generate JWT token using the utility function
        access_token = create_access_token(sub=request.username)

        return {"access_token": access_token, "token_type": "bearer", "status": "success"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")
    
    
@router.post("/uploadfile")
async def upload_file(file: UploadFile = File(...),current_user: str = Depends(get_current_user)):
    try:
        # Ensure the upload directory exists
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

        # Construct the file path
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "status": "success",
            "message": f"File '{file.filename}' has been uploaded.",
            "file_path": file_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")
    

@router.post("/deletefile")
async def delete_file(request: DeleteFileRequest, current_user: str = Depends(get_current_user)):
    try:
        # Construct the file path
        file_path = os.path.join(UPLOAD_DIR, request.filename)

        # Check if the file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File '{request.filename}' not found")

        # Delete the file
        os.remove(file_path)

        return {
            "status": "success",
            "message": f"File '{request.filename}' has been deleted."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@router.post("/forecastinput")
async def forecast_input(request: ForecastInput, current_user: str = Depends(get_current_user)):
    try:
        # Ensure the upload directory exists
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

        # Construct the file path
        file_path = os.path.join(UPLOAD_DIR, f"{request.filename}.json")

        # Save the forecast data to a file
        with open(file_path, "w") as file:
            json.dump(request.data, file)
        
        return {
            "status": "success",
            "message": f"Forecast data saved as {request.filename}.json",
            "file_path": file_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving forecast input: {str(e)}")
    
@router.post("/upload-cleaned-data-json")
async def upload_clean_date(request: UploadCleanedData, current_user: str = Depends(get_current_user)):
    try:
        # Ensure the upload directory exists
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

        file_path = os.path.join(UPLOAD_DIR, f"{request.filename}.json")

        # Prepare the data to save
        cleaned_data = {
            "granularity": request.granularity,
            "timeBucket": request.timeBucket,
            "forecastHorizon": request.forecastHorizon,
            "data": request.data
        }

        # Save the cleaned data as a JSON file
        with open(file_path, "w") as file:
            json.dump(cleaned_data, file)

        return {
            "status": "success",
            "message": f"Cleaned data saved as {request.filename}.json",
            "file_path": file_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving cleaned data: {str(e)}")

