import time
import streamlit as st
import pandas as pd
from surprise import Reader, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split
from surprise.prediction_algorithms import SVD, KNNWithMeans, KNNBasic, KNNBaseline

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="MovieLens • Movie Recommender", page_icon="🎥", layout="wide"
)

st.title("🎬 MovieLens Movie Recommender")
st.markdown("**Hybrid Recommendation System** | Powered by Collaborative Filtering")


# ====================== DATA LOADING ======================
@st.cache_data(show_spinner=False)
def load_data():
    movies_df = pd.read_csv("./Data/movies.csv")
    ratings_df = pd.read_csv("./Data/ratings.csv")
    return movies_df, ratings_df


movies_df, ratings_df = load_data()


# ====================== MODEL TRAINING ======================
@st.cache_resource(show_spinner="Training models and selecting the best one...")
def train_and_select_best_model(ratings_df):
    new_ratings_df = ratings_df.drop("timestamp", axis=1)

    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(new_ratings_df, reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=2026)

    # SVD
    svd_param_grid = {
        "n_factors": [50, 100],
        "n_epochs": [20, 50],
        "lr_all": [0.005, 0.0005],
        "reg_all": [0.02, 0.04],
    }
    svd_gs = GridSearchCV(
        algo_class=SVD, param_grid=svd_param_grid, cv=3, refit=True, measures=["rmse"]
    )
    svd_gs.fit(data)
    svd = SVD(**svd_gs.best_params["rmse"])
    svd.fit(trainset)
    svd_accuracy = accuracy.rmse(svd.test(testset), verbose=False)

    # KNNBasic
    knn_param_grid = {
        "k": [40, 50],
        "min_k": [1, 5],
        "sim_options": {"name": ["cosine", "pearson"], "user_based": [True]},
    }
    knn_basic_gs = GridSearchCV(
        algo_class=KNNBasic,
        param_grid=knn_param_grid,
        refit=True,
        measures=["rmse"],
        cv=3,
    )
    knn_basic_gs.fit(data)
    knn_basic = KNNBasic(**knn_basic_gs.best_params["rmse"])
    knn_basic.fit(trainset)
    knn_basic_accuracy = accuracy.rmse(knn_basic.test(testset), verbose=False)

    # KNNBaseline
    knn_baseline_param_grid = {
        "k": [30, 40],
        "min_k": [5, 10],
        "sim_options": {"name": ["cosine", "pearson"], "user_based": [True]},
    }
    knn_baseline_gs = GridSearchCV(
        algo_class=KNNBaseline,
        param_grid=knn_baseline_param_grid,
        refit=True,
        measures=["rmse"],
        cv=3,
    )
    knn_baseline_gs.fit(data)
    knn_baseline = KNNBaseline(**knn_baseline_gs.best_params["rmse"])
    knn_baseline.fit(trainset)
    knn_baseline_accuracy = accuracy.rmse(knn_baseline.test(testset), verbose=False)

    # KNNWithMeans
    knn_means_param_grid = {
        "k": [30, 40],
        "min_k": [5, 10],
        "sim_options": {"name": ["cosine", "pearson"], "user_based": [True]},
    }
    knn_means_gs = GridSearchCV(
        algo_class=KNNWithMeans,
        param_grid=knn_means_param_grid,
        refit=True,
        measures=["rmse"],
        cv=3,
    )
    knn_means_gs.fit(data)
    knn_with_means = KNNWithMeans(**knn_means_gs.best_params["rmse"])
    knn_with_means.fit(trainset)
    knn_with_means_accuracy = accuracy.rmse(knn_with_means.test(testset), verbose=False)

    # Select best model dynamically
    scores = {
        "SVD": (svd_gs.best_score["rmse"] + svd_accuracy) / 2,
        "KNNBasic": (knn_basic_gs.best_score["rmse"] + knn_basic_accuracy) / 2,
        "KNNBaseline": (knn_baseline_gs.best_score["rmse"] + knn_baseline_accuracy) / 2,
        "KNNWithMeans": (knn_means_gs.best_score["rmse"] + knn_with_means_accuracy) / 2,
    }
    model_objects = {
        "SVD": svd,
        "KNNBasic": knn_basic,
        "KNNBaseline": knn_baseline,
        "KNNWithMeans": knn_with_means,
    }
    rmse_results = {
        "SVD": svd_accuracy,
        "KNNBasic": knn_basic_accuracy,
        "KNNBaseline": knn_baseline_accuracy,
        "KNNWithMeans": knn_with_means_accuracy,
    }

    best_model_name = min(scores, key=lambda m: scores[m])
    best_model = model_objects[best_model_name]

    return best_model, best_model_name, scores, rmse_results, new_ratings_df


start = time.time()
best_model, best_model_name, scores, rmse_results, new_ratings_df = (
    train_and_select_best_model(ratings_df)
)
elapsed = time.time() - start

if elapsed >= 60:
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    time_str = f"{minutes}m {seconds}s"
else:
    time_str = f"{elapsed:.1f}s"

st.success(
    f"Model training complete • Best model: **{best_model_name}** "
    f"(Avg RMSE: {scores[best_model_name]:.4f}) • Training time: {time_str}"
)

# ====================== SIDEBAR ======================
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a section", ["Home", "Model Performance", "Get Recommendations", "About"]
)

# ====================== HOME PAGE ======================
if page == "Home":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Movies", f"{len(movies_df):,}")
    with col2:
        st.metric("Ratings", f"{len(ratings_df):,}")
    with col3:
        st.metric("Users", f"{ratings_df['userId'].nunique():,}")

    st.markdown(
        """
    ### Welcome to IndieFlix Recommendation Engine

    This Streamlit app is built from a collaborative filtering recommendation system.

    **What you can do:**
    - Get personalized recommendations as an **existing user** (enter User ID)
    - Get instant recommendations as a **new user** (rate movies you have seen to solve the cold-start problem)
    - View full model performance comparison
    """
    )

# ====================== MODEL PERFORMANCE PAGE ======================
elif page == "Model Performance":
    st.subheader("Model Performance Summary")

    st.metric("Best Model", best_model_name, delta="Lowest Avg RMSE")

    comparison = pd.DataFrame(
        {
            "Model": list(scores.keys()),
            "CV + Test Avg RMSE": [round(v, 4) for v in scores.values()],
            "Test RMSE": [round(rmse_results[m], 4) for m in scores.keys()],
        }
    ).sort_values("CV + Test Avg RMSE")

    st.dataframe(comparison, use_container_width=True, hide_index=True)

# ====================== RECOMMENDATIONS PAGE ======================
elif page == "Get Recommendations":
    tab1, tab2 = st.tabs(["👤 Existing User", "🆕 New User"])

    # ------------------- EXISTING USER -------------------
    with tab1:
        st.subheader("Recommendations for Existing Users")
        user_id = st.number_input(
            "Enter your User ID",
            min_value=1,
            max_value=int(ratings_df["userId"].max()),
            value=1,
            step=1,
        )

        if st.button(
            "Get My Personalized Recommendations",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Generating recommendations..."):
                all_movie_ids = new_ratings_df["movieId"].unique()
                user_movie_ids = set(
                    new_ratings_df.loc[new_ratings_df["userId"] == user_id, "movieId"]
                )

                predictions = [
                    (movie_id, best_model.predict(user_id, movie_id).est)
                    for movie_id in all_movie_ids
                    if movie_id not in user_movie_ids
                ]
                predictions.sort(key=lambda x: x[1], reverse=True)
                top_5 = predictions[:5]

                movie_id_to_title = movies_df.set_index("movieId")["title"].to_dict()
                recs = [
                    {
                        "Rank": i + 1,
                        "Movie Title": movie_id_to_title[movie_id],
                        "Predicted Rating": round(score, 3),
                    }
                    for i, (movie_id, score) in enumerate(top_5)
                ]

                st.dataframe(
                    pd.DataFrame(recs), use_container_width=True, hide_index=True
                )
                st.balloons()

    # ------------------- NEW USER -------------------
    with tab2:
        st.subheader("Recommendations for New Users")
        st.write(
            "Rate 5 movies you have seen. For each movie, select a rating and click "
            "**Rate**. If you have not seen a movie, click **Get New Samples** to "
            "replace the unrated slots with new movies."
        )

        new_user_id = int(ratings_df["userId"].max()) + 1

        # Initialise session state
        if "slots" not in st.session_state:
            initial = movies_df.sample(5).reset_index(drop=True)
            st.session_state.slots = [
                {"movie": row, "rated": False, "rating": None, "editing": False}
                for _, row in initial.iterrows()
            ]
            st.session_state.shown_movie_ids = set(initial["movieId"])

        # Get New Samples — replace unrated and editing slots
        if st.button("🔄 Get New Samples"):
            slots_to_replace = sum(
                1 for s in st.session_state.slots if not s["rated"] or s["editing"]
            )
            remaining = movies_df[
                ~movies_df["movieId"].isin(st.session_state.shown_movie_ids)
            ]
            if len(remaining) < slots_to_replace:
                st.session_state.shown_movie_ids = set(
                    s["movie"]["movieId"]
                    for s in st.session_state.slots
                    if s["rated"] and not s["editing"]
                )
                remaining = movies_df[
                    ~movies_df["movieId"].isin(st.session_state.shown_movie_ids)
                ]

            new_movies = remaining.sample(slots_to_replace).reset_index(drop=True)
            new_iter = iter(new_movies.iterrows())
            for slot in st.session_state.slots:
                if not slot["rated"] or slot["editing"]:
                    _, new_row = next(new_iter)
                    slot["movie"] = new_row
                    slot["rated"] = False
                    slot["rating"] = None
                    slot["editing"] = False
                    st.session_state.shown_movie_ids.add(new_row["movieId"])
            st.rerun()

        # Render slots
        for i, slot in enumerate(st.session_state.slots):
            movie = slot["movie"]
            if slot["rated"] and not slot["editing"]:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.success(
                        f"✅ **{movie['title']}** — You rated this: "
                        f"**{slot['rating']} / 5.0**"
                    )
                with col2:
                    if st.button("✏️ Change", key=f"change_{i}"):
                        st.session_state.slots[i]["editing"] = True
                        st.rerun()
            else:
                with st.container(border=True):
                    st.markdown(f"**{movie['title']}**")
                    rating = st.slider(
                        "Your rating",
                        min_value=0.0,
                        max_value=5.0,
                        value=(
                            float(slot["rating"]) if slot["rating"] is not None else 3.0
                        ),
                        step=0.5,
                        key=f"slider_{i}",
                    )
                    if st.button("⭐ Rate", key=f"rate_{i}"):
                        st.session_state.slots[i]["rated"] = True
                        st.session_state.slots[i]["rating"] = rating
                        st.session_state.slots[i]["editing"] = False
                        st.rerun()

        # Generate Recommendations
        num_rated = sum(
            1 for s in st.session_state.slots if s["rated"] and not s["editing"]
        )
        num_remaining = 5 - num_rated

        if st.button(
            "🎬 Generate My Recommendations", type="primary", use_container_width=True
        ):
            if num_remaining > 0:
                st.error(
                    f"Please rate {num_remaining} more "
                    f"{'movie' if num_remaining == 1 else 'movies'} before continuing."
                )
            else:
                with st.spinner(
                    "Retraining model on your ratings and generating recommendations..."
                ):
                    user_ratings = [
                        {
                            "userId": new_user_id,
                            "movieId": int(s["movie"]["movieId"]),
                            "rating": s["rating"],
                        }
                        for s in st.session_state.slots
                    ]

                    new_ratings_combined = pd.concat(
                        [new_ratings_df, pd.DataFrame(user_ratings)], ignore_index=True
                    )

                    data = Dataset.load_from_df(
                        new_ratings_combined, Reader(rating_scale=(0, 5))
                    )
                    trainset = data.build_full_trainset()
                    best_model.fit(trainset)

                    all_movie_ids = new_ratings_combined["movieId"].unique()
                    rated_ids = set(r["movieId"] for r in user_ratings)

                    predictions = [
                        (movie_id, best_model.predict(new_user_id, movie_id).est)
                        for movie_id in all_movie_ids
                        if movie_id not in rated_ids
                    ]
                    predictions.sort(key=lambda x: x[1], reverse=True)
                    top_5 = predictions[:5]

                    movie_id_to_title = movies_df.set_index("movieId")[
                        "title"
                    ].to_dict()
                    recs = [
                        {
                            "Rank": i + 1,
                            "Movie Title": movie_id_to_title[movie_id],
                            "Predicted Rating": round(score, 3),
                        }
                        for i, (movie_id, score) in enumerate(top_5)
                    ]

                    st.success("Here are your personalized recommendations!")
                    st.dataframe(
                        pd.DataFrame(recs), use_container_width=True, hide_index=True
                    )
                    st.balloons()

# ====================== ABOUT PAGE ======================
else:
    st.subheader("About This Application")
    st.markdown(
        """
    This is a Streamlit deployment of a collaborative filtering movie recommendation system.

    **Key Features:**
    - All four models (SVD, KNNBasic, KNNBaseline, KNNWithMeans) are trained and compared at startup
    - Best model is selected dynamically based on the lowest average RMSE across cross-validation and test scores
    - Existing user recommendations using collaborative filtering
    - New user cold-start handling via a checkbox-based rating interface
    - Data loaded from the local `./Data` folder

    **How to run locally:**
    ```bash
    conda install -c conda-forge streamlit
    streamlit run app.py
    ```
    """
    )
