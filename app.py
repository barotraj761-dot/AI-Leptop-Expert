
import streamlit as st
import pandas as pd
import pickle

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Laptop Expert",
    layout="wide"
)

st.title("💻 AI Laptop Expert")

# ==================================================
# LOAD MODEL
# ==================================================

with open("laptop_model.pkl", "rb") as f:

    data = pickle.load(f)

reg_model = data["reg"]
clf_model = data["clf"]
encoder_X = data["enc"]
y_clf = data["y_clf"]
df = data["df"]

# ==================================================
# CLEAN DATA
# ==================================================

df["Compny"] = df["Compny"].astype(str).str.strip()

df["Processor"] = df["Processor"].astype(str).str.strip()

# ==================================================
# BRAND IMAGES
# ==================================================

brand_images = {

    "DELL":
    "https://rukminim2.flixcart.com/image/312/312/xif0q/computer/a/e/w/-original-imahg5fuexxbwgvv.jpeg?q=70",

    "ASUS":
    "https://rukminim2.flixcart.com/image/312/312/xif0q/computer/z/y/n/-original-imahj7yfhp8hgvvs.jpeg?q=70",

    "HP":
    "https://rukminim1.flixcart.com/image/1075/1075/xif0q/computer/a/h/z/-original-imahgh9exfmtg44y.jpeg?q=90",

    "Lenovo":
    "https://rukminim1.flixcart.com/image/1075/1075/xif0q/computer/m/y/s/-original-imahgfdy579hjvu8.jpeg?q=90",

    "Acer":
    "https://rukminim2.flixcart.com/image/312/312/xif0q/computer/f/i/b/-original-imahgx8ws7ypyhmd.jpeg?q=70",

    "Apple":
    "Leptop_IMG/Apple.webp",

    "MSI":
    "Leptop_IMG/MSI2.webp",

    "Samsung":
    "Leptop_IMG/SAMSUNG.webp",

    "Infinix":
    "Leptop_IMG/infinix.webp",

    "MOTOROLA":
    "Leptop_IMG/motorola.webp",

    "Ultimus":
    "Leptop_IMG/MSI.webp",

    "Primebook":
    "Leptop_IMG/Primebook.webp",

    "WINGS":
    "Leptop_IMG/WINGS.webp"
}

# ==================================================
# PREDICTION FUNCTION
# ==================================================

def predict_laptop(brand, ram, storage):

    proc = df[df["Compny"] == brand]["Processor"].mode()[0]

    user = pd.DataFrame(
        [[brand, ram, storage, proc]],
        columns=[
            "Compny",
            "RAM_GB",
            "Storage_GB",
            "Processor"
        ]
    )

    x = encoder_X.transform(user)

    nums = reg_model.predict(x)[0]

    cats = clf_model.predict(x)[0]

    details = {}

    for i, col in enumerate(y_clf.columns):

        val = sorted(y_clf[col].unique())[int(cats[i])]

        if col == "MS_Office":

            val = (
                "Available"
                if str(val) in ["1", "1.0", "Available"]
                else "Not Available"
            )

        details[col] = val

    details["Processor"] = proc

    details["Offer"] = f"{int(nums[2])}% OFF"

    return nums, details

# ==================================================
# TABS
# ==================================================

tab1, tab2 = st.tabs([
    "💻 Prediction",
    "⚔️ Comparison"
])

# ==================================================
# TAB 1 -> PREDICTION
# ==================================================

with tab1:

    st.sidebar.header("Laptop Prediction")

    brand = st.sidebar.selectbox(
        "Company",
        sorted(df["Compny"].unique())
    )

    ram = st.sidebar.selectbox(
        "RAM",
        sorted(df["RAM_GB"].unique())
    )

    storage = st.sidebar.selectbox(
        "Storage",
        sorted(df["Storage_GB"].unique())
    )

    # ==================================================
    # PREDICT BUTTON
    # ==================================================

    if st.sidebar.button("Predict"):

        nums, details = predict_laptop(
            brand,
            ram,
            storage
        )

        col1, col2 = st.columns([1, 2])

        # ==================================================
        # IMAGE
        # ==================================================

        with col1:

            img = brand_images.get(brand)

            if img:

                st.image(
                    img,
                    width=300
                )

            else:

                st.write("No Image Available")

            st.caption(f"{brand} Laptop")

        # ==================================================
        # DETAILS
        # ==================================================

        with col2:

            st.subheader(f"{brand} Laptop Details")

            p1, p2 = st.columns(2)

            p1.metric(
                "Price",
                f"₹{int(nums[0]):,}"
            )

            p2.metric(
                "Original Price",
                f"₹{int(nums[1]):,}"
            )

            st.info(
                f"🖥️ Screen Size: {nums[3]:.1f} inch"
            )

            st.markdown("### Specifications")

            c1, c2 = st.columns(2)

            for i, (k, v) in enumerate(details.items()):

                if i % 2 == 0:

                    c1.write(f"🔹 **{k}:** {v}")

                else:

                    c2.write(f"🔹 **{k}:** {v}")

        # ==================================================
        # SIMILAR LAPTOPS
        # ==================================================

        st.markdown("---")

        st.subheader(f"🔥 Similar {brand} Laptops")

        same_brand_df = df[
            df["Compny"] == brand
        ]

        # Remove duplicate rows
        same_brand_df = same_brand_df.drop_duplicates()

        # Random 5 laptops
        similar_laptops = same_brand_df.sample(
            min(5, len(same_brand_df))
        )

        cols = st.columns(5)

        for i, (_, row) in enumerate(similar_laptops.iterrows()):

            with cols[i]:

                logo = brand_images.get(row["Compny"])

                if logo:

                    st.image(
                        logo,
                        width=120
                    )

                st.markdown(
                    f"### {row['Compny']}"
                )

                st.caption(
                    f"🧠 {row['Processor']}"
                )

                st.write(
                    f"💾 {row['RAM_GB']}GB RAM"
                )

                st.write(
                    f"💽 {row['Storage_GB']}GB Storage"
                )

                if "Processor_Tier" in df.columns:

                    st.write(
                        f"⚡ {row['Processor_Tier']}"
                    )

                st.write(
                    f"💰 ₹{row['Selling_Price']:,}"
                )

        # ==================================================
        # SMART RECOMMENDATION
        # ==================================================

        st.markdown("---")

        st.subheader("🤖 Recommended Alternatives")

        # Different Company
        recommend_df = df[
            df["Compny"] != brand
        ]

        # Similar Price Range
        recommend_df = recommend_df[
            recommend_df["Selling_Price"]
            <= nums[1] + 10000
        ]

        # Remove duplicates
        recommend_df = recommend_df.drop_duplicates()

        # Top 5
        recommend_df = recommend_df.head(5)

        cols = st.columns(5)

        for i, (_, row) in enumerate(recommend_df.iterrows()):

            with cols[i]:

                logo = brand_images.get(row["Compny"])

                if logo:

                    st.image(
                        logo,
                        width=120
                    )

                st.markdown(
                    f"### {row['Compny']}"
                )

                st.caption(
                    f"🧠 {row['Processor']}"
                )

                st.write(
                    f"💾 {row['RAM_GB']}GB RAM"
                )

                st.write(
                    f"💽 {row['Storage_GB']}GB Storage"
                )

                if "Processor_Tier" in df.columns:

                    st.write(
                        f"⚡ {row['Processor_Tier']}"
                    )

                st.write(
                    f"💰 ₹{row['Selling_Price']:,}"
                )

# ==================================================
# TAB 2 -> COMPARISON
# ==================================================

with tab2:

    st.subheader("⚔️ Compare Two Laptops")

    c1, c2 = st.columns(2)

    # ==================================================
    # LAPTOP 1
    # ==================================================

    with c1:

        st.markdown("## 💻 Laptop 1")

        brand1 = st.selectbox(
            "Brand 1",
            sorted(df["Compny"].unique()),
            key="b1"
        )

        ram1 = st.selectbox(
            "RAM 1",
            sorted(df["RAM_GB"].unique()),
            key="r1"
        )

        storage1 = st.selectbox(
            "Storage 1",
            sorted(df["Storage_GB"].unique()),
            key="s1"
        )

    # ==================================================
    # LAPTOP 2
    # ==================================================

    with c2:

        st.markdown("## 💻 Laptop 2")

        brand2 = st.selectbox(
            "Brand 2",
            sorted(df["Compny"].unique()),
            key="b2"
        )

        ram2 = st.selectbox(
            "RAM 2",
            sorted(df["RAM_GB"].unique()),
            key="r2"
        )

        storage2 = st.selectbox(
            "Storage 2",
            sorted(df["Storage_GB"].unique()),
            key="s2"
        )

    # ==================================================
    # COMPARE BUTTON
    # ==================================================

    if st.button("Compare Laptops"):

        nums1, details1 = predict_laptop(
            brand1,
            ram1,
            storage1
        )

        nums2, details2 = predict_laptop(
            brand2,
            ram2,
            storage2
        )

        col1, col2 = st.columns(2)

        # ==================================================
        # RESULT 1
        # ==================================================

        with col1:

            img1 = brand_images.get(brand1)

            if img1:

                st.image(
                    img1,
                    width=250
                )

            st.subheader(brand1)

            st.metric(
                "Price",
                f"₹{int(nums1[0]):,}"
            )

            st.info(
                f"🖥️ Screen Size: {nums1[3]:.1f} inch"
            )

            for k, v in details1.items():

                st.write(
                    f"🔹 **{k}:** {v}"
                )

        # ==================================================
        # RESULT 2
        # ==================================================

        with col2:

            img2 = brand_images.get(brand2)

            if img2:

                st.image(
                    img2,
                    width=250
                )

            st.subheader(brand2)

            st.metric(
                "Price",
                f"₹{int(nums2[0]):,}"
            )

            st.info(
                f"🖥️ Screen Size: {nums2[3]:.1f} inch"
            )

            for k, v in details2.items():

                st.write(
                    f"🔹 **{k}:** {v}"
                )
