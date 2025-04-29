# CS777TP
Lung Cancer Cell Classification based on Gene Expression Data
This project aims to process and analyze large-scale gene expression data using Apache Spark, building machine learning models to classify whether cells are lung cancer cells. We explored and implemented various classifiers to evaluate their performance on this task.

Project Objective
The core objective of the project is to develop an efficient and accurate classification model based on cell gene expression profiles for identifying lung cancer cells. This has potential value for supporting biomedical research and disease diagnosis.

Data Used
The data from this project comes from publicly available single-cell RNA sequencing data from lung cancers. The data represents how well each gene is activated in each cell. Based on those gene number activations or "expressions" we could predict if a cell is cancerous or normal. Cancerous cells have associated genes with them, and normal cells have different genes associated with them. So, theoretically, if we have gene expression data for a cell, we could predict if this cell is cancerous or normal, based on the previous data. Cells already had labels associated with them based on previous bioinformatics research. Overall, we had around 120,000 cells and around 40,000 genes in the dataset.

Project Process
We built a data processing and model training pipeline using the PySpark framework. In this pipeline, we experimented with five different classifiers:

1. Logistic Regression: A basic linear classification model, often used as a baseline.

2. Random Forest: An ensemble method based on decision trees, known for its robustness.

3. Gradient Boosted Trees Classifier (GBTC): Another powerful ensemble method that iteratively builds decision trees to improve prediction accuracy.

4. MLlib Multilayer Perceptron Classifier: Spark MLlib's built-in neural network classifier, supporting distributed training.

5. Self-built Multilayer Perceptron: A neural network classifier manually implemented using Spark RDD and NumPy, aiming to gain a deeper understanding of distributed training mechanisms.

The data processing flow includes: reading data from CSV files, label indexing, feature vector assembly, feature scaling, and dataset splitting. All these steps are performed in a Spark distributed environment to handle large-scale data.

Technical Challenges Encountered: Self-built MLP Training Failure
When attempting to train our self-built neural network classifier, we encountered significant technical difficulties, preventing the training from completing successfully. The main error logs pointed to excessive YARN space usage.

Detailed Description of Principles and Possible Causes:

1. Memory-Intensive Computations: Our Self-built MLP implementation requires using NumPy for matrix operations (forward pass, backward pass, gradient calculation) within each Spark executor. When converting Spark DataFrame/Vector data to NumPy arrays using .toArray(), this data is loaded into the executor process's memory (both JVM heap and potentially Python memory). For high-dimensional gene expression data, even a single data partition, when converted to a NumPy array, can be very large.

2. Memory Footprint of Intermediate Results: The neural network training process generates a large amount of intermediate computation results (like activation values, weighted sums, various gradients). These intermediate results also need to be stored in the executor's memory. As the number of network layers increases and partition size grows, these intermediate results accumulate and consume significant memory.

3. Memory Requirements of Batch Gradient Descent: Our implemented Batch Gradient Descent needs to accumulate gradients from all samples within a partition on each executor. Although we compute the average gradient, the gradients for each sample need to be temporarily stored or processed during the accumulation phase.

4. Broadcast Variable Overhead: Model weights and biases need to be broadcast from the Driver to all executors in each iteration. For a deep learning model with a large number of parameters, the broadcast itself can occupy significant memory on the executors.

5. Lack of Advanced Memory Optimizations: Frameworks like MLlib or TensorFlow/PyTorch perform extensive work at a lower level on memory management, data representation, and computation graph optimization to reduce memory footprint. Our manual NumPy implementation lacks these advanced optimizations.

Therefore, while our Self-built MLP might be feasible on a single machine with a small dataset, on a distributed large dataset, due to the aforementioned memory consumption and lack of optimization, the memory required by a single executor exceeded YARN's allocation limit, causing the job to fail due to memory pressure.

Project Results
Despite the Self-built MLP training not being successful, we achieved satisfactory results with the other classifiers:

1. Logistic Regression: Achieved 98% accuracy.

2. Random Forest: Achieved 98% accuracy.

3. Gradient Boosted Trees Classifier (GBTC): Achieved 98% accuracy.

4. MLlib Multilayer Perceptron Classifier: Achieved 99% accuracy.

Conclusion and Future Work
This project successfully utilized Apache Spark to process large-scale gene expression data and achieved high classification accuracy with multiple classifiers. MLlib's MLP classifier demonstrated strong capabilities on this task.

By attempting the Self-built MLP, we gained a deeper understanding of the complexity of distributed neural network training and the challenges that can be encountered in resource-constrained environments, particularly concerning memory management and optimization issues.
