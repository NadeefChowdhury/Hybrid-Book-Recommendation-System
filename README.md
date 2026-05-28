# Hybrid Book Recommendation System

## Overview

This project implements a **hybrid book recommendation system** that combines:

* **Collaborative Filtering (KNN-based)**
* **Content-Based Filtering (TF-IDF + cosine similarity among genres, authors, publishers and ratings)**

---

## Architecture

The recommendation pipeline consists of three independent modules:

### 1. Collaborative Filtering (CF)

* Based on **user-item interaction matrix**
* Uses **K-Nearest Neighbors (KNN)** to identify similar books
* Captures **collective user behavior**

### 2. Content-Based Filtering

* Uses textual and categorical metadata:

  * Genre
  * Author
  * Publisher
  * Reviews (AI generated, ignorable significance)
* Transformed using **TF-IDF vectorization**
* Computes similarity via **cosine similarity**
* Captures **intrinsic book characteristics**

### 3. Rating-Based Signals

* Incorporates:

  * Average rating (quality)
  * Rating count (popularity)
* Uses **normalized or weighted rating (IMDB-style)**

---

## Final Scoring Function

All components are combined at the final stage:

[
Score = alpha * CF + (1-alpha) * Content

Where:

* **CF** = normalized collaborative similarity
* **Content** = normalized content similarity
* **α (alpha)** controls CF vs Content balance

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* SciPy

---

## Usage

1. Prepare dataset with:

   * Book metadata
   * User ratings

2. Run preprocessing:

   * Clean data
   * Normalize rating features
   * Build content matrix
   * Build user-item matrix

3. Train models:

   * Fit KNN for CF
   * Compute content similarity

4. Generate recommendations:

```python
recommend_books("Book Title", n=5)
```

---

## Future Improvements

* Advanced ranking strategies (e.g., learning-to-rank)
* Cold-start handling
* User profiling
* Deployment as an API or web application

##
