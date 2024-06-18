import streamlit as st
import pandas as pd
from math import ceil

from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors

from Frontend import df, rating, foods
# job_titles, education_list

# Define page functions
def home():
    
    st.title("Diet Recommender based on previous interests")
    st.write(
        """
    ## Welcome to diet recommender App 🎉
    This project aims to recommend food items based on previous ratings using a trained machine learning model.
    """
    )
    st.image(
        "https://media.nutrition.org/wp-content/uploads/2017/08/myplate-705x392.jpg",
        use_column_width=True,
    )
    
    st.header("Features")
    st.write("""
    - **Dataset Description**: Describing the dataset.
    - **Enter user id**: choose between different models.
    - **Get Recommendation**: Get a suitable recommendation based on previous interest.
    """)
    
    st.header("Instructions")
    st.write("""
    1. Navigate to the 'Get Recommendation' page to insert your user id.
    2. input your user id only to get recommendations.
    3. Visit the 'Dataset' page to learn more about this project and the dataset we used to train the model.
    4. For any questions, go to the 'About us' page.
    """)


def dataset():
    st.title("Dataset")
    st.write(
        """
    ## Dataset Statistics 📊
    Here we are displaying some statistics and graphs about our dataset.
    """
    )
    st.write(df.describe())

    # Display dataset preview
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Plot some graphs
    st.write("### Number of steps Distribution")
    st.bar_chart(df.head(10000)["n_steps"])

    st.write("### Number of ingredients Distribution")
    st.bar_chart(df.head(10000)["n_ingredients"])


def recommend():
    st.title("Recommending a diet")
    st.write("## Enter the you user id to predict the salary 💼")
    st.write("Use 53932 or 57222 as example user id")
    id = st.number_input("ID", step=1)
    num_recommendation = st.number_input("Maximum cap of number of Recommendation", step=1)
    # Create predict button
    if st.button("Get Recommendation", key="recommend"):
        # Use the model to make a prediction
        recommendations = recommend_foods(id, num_recommendation) 
        display(recommendations)
        
# import streamlit as st

def display(recommended_items_ids):
    recommended_recipes = foods[foods['id'].isin(recommended_items_ids)]

    # Use Streamlit's expander and pagination
    with st.expander("Recommended Recipes"):
        # Set the number of recipes to display per page
        items_per_page = 5
        
        # Calculate the number of pages
        num_pages = (len(recommended_recipes) + items_per_page - 1) // items_per_page
        
        # Get the current page from the user
        current_page = st.number_input("Select a page", min_value=1, max_value=num_pages, value=1, step=1)
        
        # Display the recipes for the current page
        start_index = (current_page - 1) * items_per_page
        end_index = start_index + items_per_page
        for _, row in recommended_recipes.iloc[start_index:end_index].iterrows():
            st.write(f"Recipe ID: {row['id']}")
            st.write(f"Name: {row['name']}")
            st.write("-" * 100)
            
            st.write(f"Minutes: {row['minutes']}")
            st.write(f"Contributor ID: {row['contributor_id']}")
            st.write(f"Submitted: {row['submitted']}")
            st.write("-" * 100)
            st.write(f"Tags: {row['tags']}")
            st.write(f"Nutrition: {row['nutrition']}")
            st.write(f"Number of Steps: {row['n_steps']}")
            st.write("-" * 100)
            st.write(f"Steps: {row['steps']}")
            st.write("-" * 100)
            st.write(f"Description: {row['description']}")
            st.write("-" * 100)
            st.write(f"Ingredients: {row['ingredients']}")
            st.write("-" * 100)
            st.write(f"Number of Ingredients: {row['n_ingredients']}")
            st.write("-" * 100)

# Function to recommend foods for a given user
def recommend_foods(user_id, num_recommendations=5):
    # Filter out users and recipes with very few ratings
    user_threshold = 1  # Minimum number of ratings per user
    recipe_threshold = 1  # Minimum number of ratings per recipe

    # Count the number of ratings per user and recipe
    user_counts = rating['user_id'].value_counts()
    recipe_counts = rating['recipe_id'].value_counts()
    
    
    # Filter users and recipes
    filtered_users = user_counts[user_counts >= user_threshold].index
    filtered_recipes = recipe_counts[recipe_counts >= recipe_threshold].index
    filtered_ratings = rating[(rating['user_id'].isin(filtered_users)) & (rating['recipe_id'].isin(filtered_recipes))]
        
        
        
    # Get the unique user IDs and recipe IDs
    unique_user_ids = filtered_ratings['user_id'].unique()
    unique_recipe_ids = filtered_ratings['recipe_id'].unique()
    
    
    # Create dictionaries to map user IDs and recipe IDs to row and column indices
    user_id_to_row = {user_id: i for i, user_id in enumerate(unique_user_ids)}
    recipe_id_to_col = {recipe_id: i for i, recipe_id in enumerate(unique_recipe_ids)}
    
    # Initialize data for the sparse matrix
    rows = filtered_ratings['user_id'].map(user_id_to_row)
    cols = filtered_ratings['recipe_id'].map(recipe_id_to_col)
    data = filtered_ratings['rating']
    
    matrix_size = (len(unique_user_ids), len(unique_recipe_ids))
    
    # Create the sparse matrix with the calculated size
    ratings_mat_coo = coo_matrix((data, (rows, cols)), shape=matrix_size)
    
    # Convert COO matrix to CSR for efficient arithmetic operations
    ratings_mat_csr = ratings_mat_coo.tocsr()

    # Fit the NearestNeighbors model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
    model_knn.fit(ratings_mat_csr)
    
    
    
    try:
        user_index = user_id_to_row[user_id]

        distances, indices = model_knn.kneighbors(ratings_mat_csr[user_index], n_neighbors=20)

        aggregated_ratings = {}

        for i, index in enumerate(indices.flatten()):
            similar_user_ratings = ratings_mat_csr[index].indices

            weight = 1 - distances.flatten()[i]

            for item_index in similar_user_ratings:
                if item_index not in aggregated_ratings:
                    aggregated_ratings[item_index] = 0
                aggregated_ratings[item_index] += weight * (ratings_mat_csr[index, item_index] / 5.0)  # Assuming ratings are on a scale of 1 to 5

        user_rated_items = ratings_mat_csr[user_index].indices
        for item_index in user_rated_items:
            if item_index in aggregated_ratings:
                del aggregated_ratings[item_index]

        recommended_items = sorted(aggregated_ratings.keys(), key=lambda x: aggregated_ratings[x], reverse=True)[:num_recommendations]

        recommended_items_ids = [list(recipe_id_to_col.keys())[idx] for idx in recommended_items]

        return recommended_items_ids
    except KeyError:
        st.write("User ID not found in the mapping dictionary.")
        return []



            

def about_us():
    st.title('About Us')
    st.write("""
    ## Team Members

    We are a group of dedicated students working together on a project to predict annual salaries based on various factors such as age, sex, job title, and years of experience. Our team members bring diverse skills and expertise to this project, aiming to provide valuable insights and solutions.

     ### Group Members

    | **Name**          | **ID**        |
    |-------------------|---------------|
    | Rebuma Tadele     | ETS1086/13    |
    | Adugna Benti      | ETS0069/12    |
    | Nebyat Bekele     | ETS1052/13    |
    | Nebiyu Zewge      | ETS1051/13    |
    | Sabona Misgana    | ETS1114/13    |

    ________________________________________________________________

    We have combined our knowledge in data science, machine learning, and software development to create a comprehensive and user-friendly Diet Recommendation System. This project not only showcases our technical skills but also our ability to work collaboratively to achieve common goals.

    ## Our Goal
    Our goal is to develop a reliable diet recommendation model that can assist individuals in achieving their health and wellness goals. We aim to provide personalized dietary recommendations based on various factors, such as age, gender, activity level, and individual preferences. We hope that our work can contribute to a healthier and more sustainable lifestyle for our users.

    ## How It Works
    Our Diet Recommendation System uses advanced machine learning algorithms to analyze user inputs and provide tailored meal plans and dietary suggestions. Users can input their personal information, health goals, and dietary preferences, and our system will generate a customized plan to help them reach their objectives.

    ## Contact Us
    For any inquiries, feedback, or support, please feel free to contact us at [].
    """)




# Run the app
if __name__ == "__main__":
    pass
    st.write(
        "The app is running..."
    )  # This line can be omitted as Streamlit apps are run with `streamlit run script_name.py`