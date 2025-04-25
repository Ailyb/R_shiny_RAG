# app.R

# Load required R libraries
library(shiny)
library(pdftools)
library(text2vec)
library(umap)
library(keras)

# Function to process PDF and create embeddings
process_pdf <- function(pdf_path) {
  # Extract text from PDF
  text <- pdf_text(pdf_path)
  
  # Tokenize the text
  tokens <- itoken(text, preprocessor = tolower, tokenizer = word_tokenizer)
  
  # Create vocabulary
  vocab <- create_vocabulary(tokens)
  
  # Create vectorizer
  vectorizer <- vocab_vectorizer(vocab)
  
  # Create document-term matrix
  dtm <- create_dtm(tokens, vectorizer)
  
  # Create GloVe model
  glove <- GloVe$new(rank = 100, x_max = 10)
  word_vectors <- glove$fit_transform(dtm, n_iter = 10)
  
  # Convert word vectors to document vectors
  doc_vectors <- as.matrix(dtm) %*% word_vectors
  
  list(vectors = doc_vectors, text = text)
}

# Function to perform RAG
perform_rag <- function(query, pdf_data, model) {
  # Create query vector
  query_tokens <- itoken(query, preprocessor = tolower, tokenizer = word_tokenizer)
  query_dtm <- create_dtm(query_tokens, vectorizer)
  query_vector <- as.matrix(query_dtm) %*% glove$components
  
  # Calculate cosine similarity
  similarities <- sim2(pdf_data$vectors, query_vector, method = "cosine")
  
  # Get top k similar documents
  k <- 3  # Number of nearest neighbors to retrieve
  top_indices <- order(similarities, decreasing = TRUE)[1:k]
  
  # Retrieve relevant text
  relevant_text <- pdf_data$text[top_indices]
  
  # Prepare prompt for language model
  prompt <- paste("Context:", paste(relevant_text, collapse = "\n\n"), "\n\nQuestion:", query, "\n\nAnswer:")
  
  # Generate response using a simple LSTM model
  generated_text <- generate_text(model, prompt)
  
  generated_text
}

# Function to create and train a simple LSTM model
create_lstm_model <- function() {
  model <- keras_model_sequential() %>%
    layer_embedding(input_dim = 10000, output_dim = 128, input_length = 100) %>%
    layer_lstm(units = 128, return_sequences = TRUE) %>%
    layer_lstm(units = 64) %>%
    layer_dense(units = 10000, activation = "softmax")
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(),
    metrics = c("accuracy")
  )
  
  # You would need to train this model on a large text corpus
  # For demonstration purposes, we'll use random weights
  model %>% set_weights(lapply(model$get_weights(), function(x) runif(length(x))))
  
  model
}

# Function to generate text using the LSTM model
generate_text <- function(model, seed_text, max_length = 100) {
  generated <- seed_text
  for (i in 1:max_length) {
    x <- text_tokenizer$texts_to_sequences(list(generated))
    x <- pad_sequences(x, maxlen = 100)
    preds <- predict(model, x)
    next_index <- sample(1:10000, size = 1, prob = preds[1,])
    next_char <- text_tokenizer$index_word[[next_index]]
    generated <- paste0(generated, next_char)
    if (next_char == "\n" || next_char == ".") break
  }
  generated
}

# Initialize global variables
vectorizer <- NULL
glove <- NULL
text_tokenizer <- text_tokenizer(num_words = 10000)
lstm_model <- create_lstm_model()

# Define UI
ui <- fluidPage(
  titlePanel("RAG on PDF (R-only implementation)"),
  sidebarLayout(
    sidebarPanel(
      fileInput("pdf_file", "Upload PDF"),
      textInput("query", "Enter your query"),
      actionButton("submit", "Submit")
    ),
    mainPanel(
      textOutput("response")
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  # Reactive to process the uploaded PDF
  pdf_data <- reactive({
    req(input$pdf_file)
    processed_data <- process_pdf(input$pdf_file$datapath)
    
    # Update global variables
    assign("vectorizer", processed_data$vectorizer, envir = .GlobalEnv)
    assign("glove", processed_data$glove, envir = .GlobalEnv)
    
    processed_data
  })
  
  # Reactive to perform RAG when submit button is clicked
  rag_result <- eventReactive(input$submit, {
    req(input$query, pdf_data())
    perform_rag(input$query, pdf_data(), lstm_model)
  })
  
  # Output the RAG result
  output$response <- renderText({
    rag_result()
  })
}

# Run the application
shinyApp(ui = ui, server = server)
