import streamlit as st              # for front interface
import pandas as pd                 # for data manipulation
import plotly.express as px         # for data visualization
import seaborn as sns               # for data visualization
import matplotlib.pyplot as plt     # for data visualization
import openai                       # for AI prompting
import subprocess                   # for running shell script
import time                         # for some pause

# for machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# _____KEY____
def key():
    return "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"


openai.api_key = key()
# ________________


def welcome():
    st.title('Product Scraping')
    search_term = st.text_input('Which product do you want to scrape? Example: headphones')
    st.session_state.search_term = search_term

    if st.button("Scrape Data"):
        if len(search_term) != 0:
            st.write(f'Started scraping data for {search_term}...')
            scrape_products(search_term)   # running the script, robot framework and scraping products
            st.write('Scraping Completed ✅ ')

            time.sleep(2)

            # creating dashboard
            product_dashboard(search_term)
        else:
            st.write('Please enter a search term.')


def scrape_products(search_term):
    robot_script = open('amzn_product_scraping.txt', 'r').read()
    robot_script = robot_script.replace('<<<replace_string>>>', search_term)
    print(robot_script)

    # writing modified script to a temporary file
    with open('amzn_product_scraping_modified.txt', 'w') as file:
        file.write(robot_script)

    # running the script, robot framework and scraping products

    subprocess.run("./streamlit_subprocess_amzn_scraping_script.sh", shell=True)


def product_dashboard(search_term):

    def load_data():
        data = pd.read_csv('product_table.csv')
        return data

    @st.cache_data
    def filter_data():
        # Apply filters
        global df
        df = df[df['price_eur'].between(min_price, max_price)]
        return df

    # st.set_page_config(page_title='Product Dashboard')
    # Reading in the dataframe
    df = load_data()
    df = df.sort_values(by=['rating_stars', 'review_count'], ascending=[False, False])


    st.title('Product Dashboard 🛍')
    st.metric('Searched Product', search_term)
    st.metric('Products Scraped', len(df))
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric('Average Price', str(round(df['price_eur'].mean(), 2)) + "€")
    with col2:
        st.metric('Most Expensive', str(round(df['price_eur'].max(), 2)) + "€")
    with col3:
        st.metric('Cheapest', str(round(df['price_eur'].min(), 2)) + "€")

    # st.metric('Average Price', str(round(df['price_eur'].mean(), 2)) + "€")
    # st.metric('Most Expensive', str(round(df['price_eur'].max(), 2)) + "€")
    # st.metric('Cheapest', str(round(df['price_eur'].min(), 2)) + "€")

    # product_name = st.text_input("Enter the product you're searching for: ")

    # Product Table
    st.header('Product Table')

    # _________________ Sidebar ___________________
    # will be used for filtering

    st.sidebar.header('Filters')
    min_price = st.sidebar.number_input('Min Price', min_value=0, key=1, on_change=filter_data)
    max_price = st.sidebar.number_input('Max Price', min_value=0, key=2, on_change=filter_data)
    min_review_count = st.sidebar.number_input('Min Review Count', min_value=0, key=3)
    min_stars = st.sidebar.number_input('Min Stars', min_value=1, max_value=5)
    max_stars = st.sidebar.number_input('Max Stars', min_value=1, max_value=5)
    if st.button('Update Data'):
        st.experimental_rerun()

    st.dataframe(df)

    # _____________________________________________

    # _________________ AI Section ___________________
    st.subheader('Trends in Product Data')

    # DataFrame for AI prompts
    first_30_products = df.sort_values(by='review_count', ascending=False)[:30].to_string()
    st.write(ask_ai_for_trends(first_30_products))

    ai_user_prompt = st.text_input("Ask AI anything about your scraped products (first 30 rows)")
    if len(ai_user_prompt) != 0:
        st.write(ask_ai_anything(first_30_products, ai_user_prompt))
    # _________________________________________________

    # ________________Distributions ____________________
    st.header('Price Distribution')
    fig = px.histogram(df, x='price_eur')
    st.plotly_chart(fig, use_container_width=True)

    # ________________ Scatter Plot ____________________
    st.header('Correlation Scatter Plot')
    fig_df = df.copy()
    fig_df['point_color'] = pd.cut(df['review_count'], \
                                   bins=[0, 200, 1000, df['review_count'].max()], \
                                   labels=['red', 'orange', 'green'])
    color_scale = {'red': 'red', 'orange': 'orange', 'green': 'green'}
    color_scale = {'red': '#FF5733', 'orange': '#FFA500', 'green': '#00FF00'}
    fig2 = px.scatter(fig_df, x='review_count', y='rating_stars', color='point_color', color_discrete_map=color_scale)
    st.plotly_chart(fig2, use_container_width=True)
    # df[(df['price_eur'].between(20, 80)) & (df.rating_stars >= 3.7)]

    # ______________MACHINE LEARNING _________
    perform_machine_learning(df)


def perform_machine_learning(df):
    # Goal is to predict the price based on the review count and rating stars of a product

    st.header('Machine Learning Performace')

    # subset dataframe for machine learning
    df_for_ml = df[['price_eur', 'review_count', 'rating_stars']]

    # feature and target variable selection
    X = df_for_ml[['review_count', 'rating_stars']]
    y = df_for_ml['price_eur']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # LINEAR REGRESSION
    st.subheader('Model: Linear Regression')
    mse_linreg, r2_linreg, y_test, y_pred_linreg = perform_linear_regression(X_train, X_test, y_train, y_test)
    st.metric('Mean Squared Error', round(mse_linreg, 4))
    st.metric('R-Squared', round(r2_linreg, 4))

    # predicted vs actual price plot for linear regressionj
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred_linreg)
    plt.title('Predicted vs Actual Prices - GBM Model')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red',
             lw=2)
    st.pyplot(plt)

    # residual plot
    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_test, y=y_pred_linreg, lowess=True, color="g")
    plt.title('Residuals of Predictions - GBM Model')
    plt.xlabel('Actual Prices')
    plt.ylabel('Residuals')
    st.pyplot(plt)
    st.write('')

    # GRADIENT BOOSTING
    st.subheader('Model: Gradient Boosting')
    mse_gbm, r2_gbm, y_test, y_pred_gbm = perform_gradient_boosting(X_train, X_test, y_train, y_test)
    st.metric('Mean Squared Error', round(mse_gbm, 4))
    st.metric('R-Squared', round(r2_gbm, 4))

    # predicted vs actual price plot for gradient boosting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred_gbm)
    plt.title('Predicted vs Actual Prices - GBM Model')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red',
             lw=2)
    st.pyplot(plt)

    # residual plot
    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_test, y=y_pred_gbm, lowess=True, color="g")
    plt.title('Residuals of Predictions - GBM Model')
    plt.xlabel('Actual Prices')
    plt.ylabel('Residuals')
    st.pyplot(plt)


def perform_linear_regression(X_train, X_test, y_train, y_test):
    """Goal is to predict the price based on the review count and rating stars of a product"""
    linreg_model = LinearRegression()

    # Train the model
    linreg_model.fit(X_train, y_train)

    # Predict prices
    y_pred_linreg = linreg_model.predict(X_test)

    # Evaluate the model
    mse_linreg = mean_squared_error(y_test, y_pred_linreg)
    r2_linreg = r2_score(y_test, y_pred_linreg)

    return [mse_linreg, r2_linreg, y_test, y_pred_linreg]


def perform_gradient_boosting(X_train, X_test, y_train, y_test):
    """Goal is to predict the price based on the review count and rating stars of a product"""

    gbm_model = GradientBoostingRegressor(random_state=0)
    gbm_model.fit(X_train, y_train)         # training model
    y_pred_gbm = gbm_model.predict(X_test)  # predicting prices

    # model evaluation
    mse_gbm = mean_squared_error(y_test, y_pred_gbm)
    r2_gbm = r2_score(y_test, y_pred_gbm)

    return [mse_gbm, r2_gbm, y_test, y_pred_gbm]


def ask_ai_for_trends(product_data_string):
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "user", "content": f"What are some noticeable trends in the following product data:\n{product_data_string}\n-Provide 3 good points please. Output with bullet points."}
      ]
    )
    return completion.choices[0].message.content


def ask_ai_anything(product_data_string, user_prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": f"{user_prompt}:\nproduct data:{product_data_string}"}
        ]
    )
    return completion.choices[0].message.content


def load_data():
    data = pd.read_csv('product_table.csv')
    return data


# ______START______
welcome()


# # Dashboard Only
# term = 'usb c cable'
#
# product_dashboard(term)
#











# @st.cache_data
# def load_data():
#     data = pd.read_csv('product_table.csv')
#     return data
#
#
#
# @st.cache_data
# def filter_data():
#     # Apply filters
#     global df
#     df = df[df['price_eur'].between(min_price, max_price)]
#     return df
# @st.cache_data
# def filter_data():
#     # Apply filters
#     global df
#     df = df[df['price_eur'].between(min_price, max_price)]
#
#     return df
# @st.cache_data

# if __name__ == '__main__':
#     main()



# ____ INITIAL WELCOME ___

# def main():
#     st.title('Product Scraping')
#     print('Here')
#     # Check if the search input has been displayed
#     if 'search_term' not in st.session_state:
#         # search_term = st.text_input('Which product do you want to scrape? Example: headphones')
#         st.session_state.search_term = st.text_input('Which product do you want to scrape? Example: headphones')
#         st.write(st.session_state.search_term)





# ____ Session State ___
# if 'search_term' not in st.session_state:
#     st.write('Not in session')
#     st.session_state["search_term"] = st.text_input('Which product do you want to scrape? Example: headphones')
# st.session_state.search_term = 'Hello'
# st.write(st.session_state.search_term)

# if search_term:
#     # _____________
#     st.write(f"Product: {search_term}")
#
#     df = load_data()
#     df = df.sort_values(by=['rating_stars', 'review_count'], ascending=[False, False])
#
#     st.title('Product Dashboard')
#     st.metric('Products Scraped', len(df))
#     st.metric('Average Price', str(round(df['price_eur'].mean(), 2)) + "€")
#     st.metric('Most Expensive', str(round(df['price_eur'].max(), 2)) + "€")
#     st.metric('Cheapest', str(round(df['price_eur'].min(), 2)) + "€")
#
#     # product_name = st.text_input("Enter the product you're searching for: ")
#
#
#     # Product Table
#     st.header('Product Table')
#
#
#     # Sidebar
#     st.sidebar.header('Filters')
#     min_price = st.sidebar.number_input('Min Price', min_value=0, key=1, on_change=filter_data)                # , on_change=filter_data
#     max_price = st.sidebar.number_input('Max Price', min_value=0, key=2, on_change=filter_data)
#     min_review_count = st.sidebar.number_input('Min Review Count', min_value=0, key=3)
#     min_stars = st.sidebar.number_input('Min Stars', min_value=1, max_value=5)
#     max_stars = st.sidebar.number_input('Max Stars', min_value=1, max_value=5)
#     if st.button('Update Data'):
#         st.experimental_rerun()
#
#     st.dataframe(df)
#
#
#     # ____ AI Section ____
#     st.subheader('Trends in Product Data')
#
#     # DataFrame for AI prompts
#     first_30_products = df.sort_values(by='review_count', ascending=False)[:30].to_string()
#
#
#     st.write(ask_ai_for_trends(first_30_products))
#
#
#     ai_user_prompt = st.text_input("Ask AI anything about your scraped products (first 30 rows)")
#     if len(ai_user_prompt) != 0:
#         st.write(ask_ai_anything(first_30_products, ai_user_prompt))
#
#
#     # Distributions
#     st.header('Price Distribution')
#     fig = px.histogram(df, x='price_eur')
#     st.plotly_chart(fig, use_container_width=True)
#
#     fig_df = df.copy()
#     fig_df['point_color'] = pd.cut(df['review_count'], \
#                                    bins=[0, 200, 1000, df['review_count'].max()], \
#                                    labels=['red', 'orange', 'green'])
#     color_scale = {'red': 'red', 'orange': 'orange', 'green': 'green'}
#     color_scale = {'red': '#FF5733', 'orange': '#FFA500', 'green': '#00FF00'}
#     fig2 = px.scatter(fig_df, x='review_count', y='rating_stars', color='point_color', color_discrete_map=color_scale)
#     # Update the style and layout of the plot
#
#     st.plotly_chart(fig2, use_container_width=True)
#     # df[(df['price_eur'].between(20, 80)) & (df.rating_stars >= 3.7)]
#
#     if st.button('Perform Machine Learning'):
#         st.write('Linear Regression Performance')
#         st.write('Neural Network Performance')
#
#
# # Filter dataframe based on sidebar inputs
# # df = df[df['review_count'] >= min_review_count]
# # df = df[df['rating_stars'].between(min_stars, max_stars)]
