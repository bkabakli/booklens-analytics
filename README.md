# 📚 BookLens Analytics

Decision support dashboard for a B2B book distributor, built with Streamlit for the ISL 533 Data Mining course.

**Live dashboard:** https://booklens-analytics.streamlit.app/

## Pages
- **Dataset Overview:** attributes, attribute types and first observations
- **Data Preprocessing:** duplicates, noisy data, missing values, distribution and IQR outlier analysis
- **Sales Overview:** KPIs, monthly trend, top categories and cities
- **Sales Forecast:** Linear Regression vs SVR, back-tested on the last 4 weeks
- **Customer Segmentation:** RFM analysis
- **ABC & Best Sellers:** ABC (Pareto) analysis and best-selling products

## Files
- `streamlit_app.py` – application source code
- `requirements.txt` – required Python libraries
- `B2B_Transaction_Data.parquet` – dataset (the original Excel file converted to Parquet for faster loading and to stay within GitHub's file size limit; the content is unchanged)

## Run locally
    pip install -r requirements.txt
    streamlit run streamlit_app.py
