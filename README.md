# RAG on PDF - R-only Implementation

This Shiny application demonstrates a basic Retrieval-Augmented Generation (RAG) system entirely implemented in R, without relying on Python or the reticulate package.

## Prerequisites

- R (version 4.0 or later)
- RStudio (recommended for development)

## Required R Packages

- shiny
- pdftools
- text2vec
- umap
- keras

You can install these packages using the following command in R:

```r
install.packages(c("shiny", "pdftools", "text2vec", "umap", "keras"))
```

## How to Run
1. Clone this repository or download the app.R file.
2. Open RStudio and set the working directory to the location of app.R.
3. Run the following command in R:
```
shiny::runApp("app.R")
```
The application should now be running in your default web browser.
## Usage
1. Upload a PDF file using the "Upload PDF" button.
2. Enter your query in the "Enter your query" text input field.
3. Click the "Submit" button to see the RAG response.

## Limitations
  - This implementation uses a simple cosine similarity calculation for document retrieval, which may be slow for large document sets.
  - The language model is a basic LSTM implemented with Keras and uses random weights. It will not produce meaningful or coherent responses without proper training.
  - The text embedding is performed using GloVe from the text2vec package, which may not be as effective as more modern embedding techniques.

## Potential Improvements
  - Implement a more efficient nearest neighbor search algorithm.
  - Train the LSTM model on a large, domain-specific text corpus.
  - Use more advanced embedding techniques or pre-trained models.
  - Improve error handling and user feedback in the Shiny interface.
  - Optimize the application for larger documents and higher query volumes.

## Note
This is a proof-of-concept implementation and is not intended for production use. For a more robust RAG system, consider using specialized libraries and tools, including those that integrate with Python.

## License
This project is licensed under the MIT License.


