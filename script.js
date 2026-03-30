// document.querySelectorAll('.details-icon').forEach(icon => {
//   icon.addEventListener('click', () => {
//     const cardId = icon.getAttribute('data-card');
//     document.getElementById(cardId).style.display = 'flex';
//   });
// });

// document.querySelectorAll('.close-btn').forEach(btn => {
//   btn.addEventListener('click', () => {
//     btn.closest('.card-overlay').style.display = 'none';
//   });
// });

// document.querySelectorAll('.card-overlay').forEach(overlay => {
//   overlay.addEventListener('click', event => {
//     // Close only if the click was outside the actual card (i.e., on the overlay)
//     if (!event.target.closest('.card')) {
//       overlay.style.display = 'none';
//     }
//   });
// });



// Mobile Menu Toggle
const menuToggle = document.getElementById('menu-toggle');
const navLinks = document.getElementById('nav-links');

menuToggle.addEventListener('click', () => {
  navLinks.classList.toggle('active');

  // Change icon
  if (menuToggle.textContent === '☰') {
    menuToggle.textContent = '✖';
  } else {
    menuToggle.textContent = '☰';
  }
});

// Dark Mode Toggle
const toggleBtn = document.getElementById('theme-toggle');

toggleBtn.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');

    // Optional: change icon
    if (document.body.classList.contains('dark-mode')) {
        toggleBtn.textContent = '☀️';
    } else {
        toggleBtn.textContent = '🌙';
    }
});


// For Experience Section
// Open Modal
function openModal(id) {
  document.getElementById(id).style.display = 'flex';
}

// Close Modal
function closeModal(id) {
  document.getElementById(id).style.display = 'none';
}

// Close Modal When Clicking Outside the Modal Content
document.addEventListener('click', function (event) {
  const modals = document.querySelectorAll('.modal-overlay');
  modals.forEach(modal => {
    if (event.target === modal) {
      modal.style.display = 'none';
    }
  });
});

// Expand bullets inline on click
document.querySelectorAll('.see-more').forEach(link => {
  link.addEventListener('click', function(e) {
    e.preventDefault(); // prevent page jump
    const parentLi = this.parentElement;
    const moreText = parentLi.querySelector('.more-text');
    if (moreText) {
      moreText.style.display = 'inline';
      this.style.display = 'none'; // hide the link
    }
  });
});


// For Projects Section
document.addEventListener('DOMContentLoaded', () => {
  const projects = [
    {
      title: "California Housing Price Prediction",
      category: "other",
      image: "assets/images/california.jpg",
      description:
        "An end-to-end machine learning regression project that predicts California housing prices using demographic and geographic data. The project walks through the entire ML pipeline including data cleaning, feature engineering, model training, and hyperparameter tuning with Random Forests.",
      goal:
        "To develop a high-performing regression model that can predict housing values based on various socioeconomic features, assisting in real estate decision-making and analysis.",
      skills: [
        "Performed data cleaning, handling missing values, and stratified sampling",
        "Engineered custom features and built preprocessing pipelines",
        "Visualized data distributions, correlations, and geospatial patterns",
        "Trained a Random Forest Regressor and fine-tuned it using GridSearchCV",
        "Evaluated model performance using RMSE, R², and cross-validation",
        "Implemented an end-to-end pipeline from preprocessing to prediction"
      ],
      tools: {
        languages: ["Python"],
        aiModel: "RandomForestRegressor with GridSearchCV",
        libraries: [
          "pandas",
          "numpy",
          "scikit-learn",
          "matplotlib"
        ],
        hosting: "Local Jupyter Notebook",
        media: "Data visualizations, hyperparameter tuning results"
      },
      github: "https://github.com/ptrishita/End-to-End-ML-Project---California-Housing"
    },
    {
      title: "Bank Customer Churn Prediction",
      category: "other",
      image: "assets/images/churn.png",
      description:
        "A predictive analytics project designed to identify which bank customers are likely to churn (leave the bank), using an Artificial Neural Network (ANN). By analyzing demographic, account, and behavioral data, the model helps banks proactively retain high-risk customers.",
      goal:
        "To develop a robust ANN-based classification model that accurately predicts customer churn, enabling banks to implement targeted retention strategies and reduce customer attrition.",
      skills: [
        "Performed exploratory data analysis (EDA) on customer demographic and account features",
        "Engineered features and handled data cleaning/preprocessing (e.g., encoding, scaling)",
        "Built and compiled an ANN using Keras/TensorFlow for binary classification",
        "Trained the model with appropriate validation splits and early stopping",
        "Evaluated model performance using metrics like accuracy, ROC-AUC, precision, and recall",
        "Visualized training history, confusion matrix, and ROC curves"
      ],
      tools: {
        languages: ["Python"],
        aiModel: "Artificial Neural Network (Keras/TensorFlow)",
        libraries: [
          "pandas",
          "numpy",
          "scikit-learn",
          "keras",
          "tensorflow",
          "matplotlib",
          "seaborn"
        ],
        hosting: "Local Jupyter Notebook",
        media: "EDA plots, training history charts, confusion matrices, ROC curves"
      },
      github: "https://github.com/ptrishita/Churn-Predictions-by-ANN"
    },
    {
      title: "Face Recognition Attendance System",
      category: "other",
      image: "assets/images/face_recognition.jpg",
      description:
        "An AI-powered attendance management application that uses facial recognition to automate and streamline attendance tracking. The system captures and processes live webcam images or pre-collected datasets to recognize individuals, mark attendance in real time, and maintain secure records.",
      goal:
        "To build an efficient and contactless attendance system using computer vision techniques—leveraging OpenCV’s face detection and recognition algorithms—to reduce manual effort and prevent fraud in attendance tracking.",
      skills: [
        "Implemented face detection with Haar Cascade (OpenCV)",
        "Extracted facial embeddings and compared them for recognition",
        "Developed real-time face capture and recognition via webcam",
        "Built a GUI interface using Tkinter for user interaction",
        "Stored attendance logs in CSV/MySQL and generated downloadable reports",
        "Trained and evaluated recognition accuracy; mitigated false matches"
      ],
      tools: {
        languages: ["Python"],
        aiModel: "Haar Cascade + FaceNet / LBPH algorithms",
        libraries: [
          "OpenCV",
          "face_recognition",
          "Tkinter",
          "Pandas",
          "NumPy",
          "MySQL Connector / CSV I/O"
        ],
        hosting: "Local desktop application (Python/Tkinter)",
        media: "GUI screenshots, real-time webcam feed, attendance CSV reports"
      },
      github: "https://github.com/ptrishita/Attendance-Management-System-using-Face-Recognition"
    },
    {
      title: "Restaurant Reviews Sentiment Analysis",
      category: "other",
      image: "assets/images/sentiment.png",
      description:
        "This project analyzes customer reviews of restaurants to determine sentiment polarity—positive, neutral, or negative. By leveraging natural language processing (NLP) techniques and classic machine learning models, it transforms unstructured text data into actionable insights about customer satisfaction.",
      goal:
        "To build a robust sentiment classification system that accurately categorizes restaurant reviews, enabling businesses to understand customer perceptions and improve service quality.",
      skills: [
        "Processed and cleaned textual review data",
        "Performed feature extraction using techniques like Bag‑of‑Words and TF‑IDF",
        "Trained and evaluated sentiment classification models (e.g., Logistic Regression, Naive Bayes)",
        "Visualized performance with confusion matrices, accuracy, and classification reports",
        "Explored NLP best practices such as tokenization, stop word removal, and lemmatization"
      ],
      tools: {
        languages: ["Python"],
        aiModel: "Logistic Regression / Naive Bayes classifiers",
        libraries: [
          "pandas",
          "sklearn",
          "nltk",
          "matplotlib",
          "seaborn"
        ],
        hosting: "Local Jupyter Notebook",
        media: "Visualizations include word clouds, confusion matrices, and classifier performance metrics"
      },
      github: "https://github.com/ptrishita/Restaurant-Reviews-Sentiment-Analysis"
    },
    {
      title: "Bike Sharing Algorithm",
      category: "other",
      image: "assets/images/bike.jpg",
      description:
        "Bike Sharing Algorithm is a regression-based machine learning project developed to predict bike demand based on time, weather, and seasonal data. The model processes and analyzes historical bike-sharing data to forecast the number of bike rentals, offering valuable insights for resource planning and optimization.",
      goal:
        "To build a predictive model that estimates hourly bike demand using historical and environmental data, enabling bike-sharing services to allocate resources more efficiently.",
      skills: [
        "Performed time-series feature extraction from timestamps (hour, weekday, month, year)",
        "Conducted data visualization and EDA including histograms, heatmaps, and pairplots",
        "Preprocessed data using normalization and correlation analysis",
        "Developed and evaluated multiple regression models (Linear Regression and Random Forest Regressor)",
        "Visualized model performance using actual vs predicted plots and residual analysis",
        "Compared model metrics (MAE, MSE, R²) for performance tuning"
      ],
      tools: {
        languages: ["Python"],
        aiModel: "Linear Regression, Random Forest Regressor",
        libraries: [
          "pandas",
          "matplotlib",
          "seaborn",
          "scikit-learn"
        ],
        hosting: "Local Jupyter Notebook / Cloud Notebook (Kaggle or Colab)",
        media: "Regression plots, EDA visualizations (histograms, heatmaps, residual plots)"
      },
      github: "https://github.com/ptrishita/Bike_Sharing_Algorithm"
    },
    {
      title: "Loan Approval Classifier",
      category: "other",
      image: "assets/images/loan.jpg",
      description:
        "Loan Approval Classifier is a machine learning project designed to predict loan approval outcomes using applicant demographic and financial data. The project features extensive data preprocessing, exploratory data analysis (EDA), feature engineering, and model optimization. A Random Forest Classifier is used to build the predictive model, with hyperparameter tuning to enhance performance.",
      goal:
        "To accurately predict whether a loan application should be approved by training a classification model on structured financial and demographic data, helping financial institutions automate and enhance decision-making processes.",
      skills: [
        "Performed data cleaning and preprocessing on a real-world loan dataset",
        "Conducted exploratory data analysis (EDA) with visualizations for both numerical and categorical features",
        "Built preprocessing pipelines using Scikit-learn for encoding and scaling",
        "Developed a machine learning model using Random Forest Classifier",
        "Evaluated model performance using accuracy, confusion matrix, and classification report",
        "Optimized model performance through hyperparameter tuning using GridSearchCV"
      ],
      tools: {
        languages: ["Python"],
        aiModel: "Random Forest Classifier",
        libraries: [
          "pandas",
          "numpy",
          "matplotlib",
          "seaborn",
          "scikit-learn"
        ],
        hosting: "Local Jupyter Notebook / Kaggle Notebook environment",
        media: "EDA plots (histograms, boxplots, heatmaps), model evaluation metrics"
      },
      github: "https://github.com/ptrishita/Loan_Approval_Classifier"
    },
    {
      title: "PEFT for Text Summarization",
      category: "other",
      image: "assets/images/peft.png",
      description:
        "This project implements text summarization using the BART model fine-tuned with Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically LoRA (Low-Rank Adaptation). Leveraging the CNN/DailyMail dataset, it demonstrates how to significantly reduce the number of trainable parameters while maintaining performance. The solution is built using Hugging Face Transformers, Datasets, and the PEFT library.",
      goal:
        "To explore efficient fine-tuning strategies for large language models by applying LoRA to a pre-trained BART model for the task of abstractive text summarization, achieving competitive results with minimal training resources.",
      skills: [
        "Applied PEFT using LoRA configuration for text summarization",
        "Fine-tuned the BART-large model on the CNN/DailyMail dataset",
        "Preprocessed and tokenized large-scale datasets using Hugging Face Datasets",
        "Implemented summarization logic and training using the Transformers Trainer API",
        "Visualized and evaluated summarization outputs for qualitative validation",
        "Worked in a resource-constrained environment using only necessary trainable parameters"
      ],
      tools: {
        languages: ["Python"],
        aiModel: "facebook/bart-large-cnn with LoRA (PEFT)",
        libraries: [
          "transformers",
          "peft",
          "datasets",
          "evaluate",
          "torch",
          "accelerate"
        ],
        hosting: "Executed in a Kaggle notebook (adaptable to Colab or local)",
        media: "Text summarization input/output; evaluation via console print"
      },
      github: "https://github.com/ptrishita/PEFT_for_Text_Summarization"
    },
    {
      title: "AI-Powered Chatbot using Google Gemini",
      category: "other",
      image: "assets/images/chatbot.jpg",
      description:
        "An interactive web-based chatbot application powered by Google's Gemini 2.0 Flash model. This lightweight Streamlit app enables real-time, natural language conversations by integrating with the GenerativeAI API. It simulates an intelligent assistant capable of understanding and responding to a wide range of user queries.",
      goal:
        "To create a conversational AI experience that showcases the capabilities of the Gemini model in generating coherent and helpful responses, making AI-based assistance accessible through a simple and user-friendly interface.",
      skills: [
        "Built a real-time chatbot using Streamlit",
        "Integrated Google Gemini 2.0 Flash model via GenerativeAI API",
        "Implemented modular Python scripting using `app.py` and `chatbot.py`",
        "Handled secure API key storage using environment variables and `dotenv`",
        "Created responsive and minimalistic user interface with Streamlit widgets",
        "Implemented error handling for robust user interaction"
      ],
      tools: {
        languages: ["Python"],
        aiModel: "Google Gemini 2.0 Flash via GenerativeAI API",
        libraries: ["google-generativeai", "streamlit", "dotenv"],
        hosting: "Local/Cloud deployment using Streamlit",
        media: "Text-based input/output; emoji icons for enhanced UX"
      },
      github: "https://github.com/ptrishita/AI-Powered_Chatbot_using_Google_Gemini"
    },
    {
      title: "Winning Space Race (IBM Data Science Capstone)",
      category: "featured",
      image: "assets/images/space.png",
      description:
        "A comprehensive capstone project from the IBM Data Science specialization that explores SpaceX’s launch performance. The project involves data collection, wrangling, visualization, machine learning, and interactive dashboarding to analyze and predict launch outcomes, helping SpaceX improve mission success rates.",
      goal:
        "To apply the full data science lifecycle—from data acquisition to deployment—toward understanding and optimizing SpaceX launch data, enabling better decision-making through data-driven insights.",
      skills: [
        "Collected data via REST API (SpaceX API) and web scraping (Wikipedia)",
        "Performed data wrangling using Pandas and cleaned JSON and HTML sources",
        "Conducted EDA using Matplotlib, Seaborn, and Plotly for interactive charts",
        "Created SQL queries for structured data exploration in Jupyter Notebooks",
        "Built an interactive dashboard using Plotly Dash to visualize launch insights",
        "Implemented a classification model to predict successful launches using machine learning",
        "Mapped geographic launch patterns using Folium"
      ],
      tools: {
        languages: ["Python", "SQL"],
        aiModel: "Logistic Regression / Classification Models for launch success prediction",
        libraries: [
          "pandas",
          "numpy",
          "matplotlib",
          "seaborn",
          "plotly",
          "dash",
          "scikit-learn",
          "folium",
          "BeautifulSoup",
          "requests"
        ],
        hosting: "Local Jupyter Notebooks, Dash Web App (Python)",
        media: "Dash dashboards, Folium maps, SQL queries, visual analytics"
      },
      github: "https://github.com/ptrishita/IBM_Data_Science_Assignment/tree/main/Applied%20Data%20Science%20Capstone"
    },
    {
      title: "CareerCraft: ATS-Optimized Resume Analyzer using Gemini Model",
      category: "featured",
      image: "assets/images/careercraft.png",
      description:
        "CareerCraft is a cutting-edge resume analysis web application that leverages Google's Gemini 2.0 Flash generative AI model to assess resumes against job descriptions using ATS (Applicant Tracking System) logic. It helps job seekers optimize their resumes by identifying missing keywords, estimating match percentages, and generating profile summaries, all in real-time.",
      goal:
        "To simplify and enhance the job application process by providing users with intelligent, automated insights on resume optimization, ensuring better alignment with job descriptions and improving visibility through ATS systems.",
      skills: [
        "Applied prompt engineering for resume analysis and keyword matching",
        "Integrated Google Generative AI (Gemini) via API",
        "Built an interactive UI/UX with Streamlit for real-time user feedback",
        "Extracted and processed PDF resume data using PyPDF2",
        "Implemented secure API key handling with environment variables",
        "Designed structured outputs including match percentage, missing keywords, and profile summaries",
        "Developed a feature-rich, visually appealing web app with rich media and interactive elements"
      ],
      tools: {
        languages: ["Python"],
        aiModel: "Google Gemini 2.0 Flash via GenerativeAI API",
        libraries: ["Streamlit", "streamlit_extras", "PyPDF2", "PIL", "dotenv"],
        hosting: "Local/Cloud Streamlit deployment (potential for GCP or Hugging Face Spaces)",
        media: "PIL for images, PDF parsing for resume text extraction"
      },
      github: "https://github.com/ptrishita/CareerCraft-ATS-Optimized-Resume-Analyzer-using-Gemini-Model",
      demo: "https://youtu.be/E0cgbWYTR9k"
    },
    {
      title: "Machine Learning Approach for Employee Performance Prediction",
      category: "other",
      image: "assets/images/performance_evaluation.webp",
      description:
        "A machine learning–based web application that predicts employee productivity using historical workplace data such as working hours, idle time, overtime, and incentives, enabling data-driven decision-making.",
      goal:
        "To predict employee performance using regression models and provide actionable insights for talent management, performance improvement, and resource allocation.",
      skills: [
        "Applied data preprocessing and feature engineering on real-world datasets",
        "Implemented and compared multiple regression algorithms (Linear Regression, Random Forest, XGBoost)",
        "Evaluated model performance using MSE, MAE, and R² metrics",
        "Selected and deployed the best-performing model (XGBoost)",
        "Developed a Flask-based web application for prediction",
        "Handled categorical encoding and data cleaning techniques"
      ],
      tools: {
        languages: ["Python"],
        aiModel: "Linear Regression, Random Forest Regressor, XGBoost Regressor",
        libraries: ["Pandas", "NumPy", "Matplotlib", "Seaborn", "Scikit-learn"],
        hosting: "Flask web app",
        media: "Tabular dataset (garments_worker_productivity.csv)"
      },
      github: "https://github.com/ptrishita/Machine-Learning-Approach-For-Employee-Performance-Prediction",
      demo: "https://youtu.be/lbORl-9Q27w"
    },
    {
      title: "Designing an Autonomous Learning Agent with Checkpoint Verification and Feynman Pedagogy",
      category: "other",
      image: "assets/images/learning_agent.png",
      description:
        "An AI-powered autonomous learning system that guides users through structured learning pathways using checkpoint-based verification and the Feynman technique, ensuring mastery-based progression and adaptive concept understanding.",
      goal:
        "To create a personalized AI tutor that enforces conceptual clarity by verifying understanding at each stage and adapting explanations using simplified, analogy-based teaching methods.",
      skills: [
        "Designed stateful AI workflows using LangGraph for structured learning",
        "Implemented checkpoint-based learning with quantitative verification",
        "Applied Feynman technique for adaptive explanation and concept simplification",
        "Built multi-agent research and tutoring system using LangChain",
        "Integrated LLM APIs with external tools for dynamic content generation",
        "Developed modular architecture with conditional routing and feedback loops"
      ],
      tools: {
        languages: ["Python"],
        aiModel: "LLM APIs (Gemini, Groq)",
        libraries: ["LangGraph", "LangChain", "python-dotenv"],
        hosting: "Local / cloud-based deployment",
        media: "Text input, web search data, and knowledge base embeddings"
      },
      github: "https://github.com/springboardmentor425/Designing-an-Autonomous-Learning-Agent-with-Checkpoint-Verification-and-Feynman-Pedagogy-/tree/trishita"
    },
    {
      title: "Smart Nutrition & Wellness App using Gemini Flash",
      category: "featured",
      image: "assets/images/nutrition.jpg",
      description:
        "An AI-powered web application that analyzes food items, diet reports (PDFs), and food images to provide personalized nutrition insights and wellness recommendations using Gemini Flash.",
      goal:
        "To deliver personalized, AI-driven nutrition guidance and diet planning by analyzing user inputs, food data, and health reports in real time.",
      skills: [
        "Applied prompt engineering for nutrition analysis and personalized recommendations",
        "Integrated Google Gemini Flash for text, image, and PDF-based analysis",
        "Built an interactive Streamlit web application with multiple features",
        "Processed and extracted insights from PDF reports using PyPDF2",
        "Implemented image-based food analysis using PIL",
        "Designed modular and scalable UI for real-time user interaction"
      ],
      tools: {
        languages: ["Python"],
        aiModel: "Google Gemini Flash",
        libraries: ["Streamlit", "streamlit_extras", "PyPDF2", "PIL", "dotenv"],
        hosting: "Streamlit web app",
        media: "Text input, PDF reports, and food images"
      },
      github: "https://github.com/ptrishita/Smart-Nutrition-Wellness-App-using-Gemini-Flash"
    },
    {
      title: "ZenFlow: Enlightened Yoga Pose Classification Via Transfer Learning",
      category: "featured",
      image: "assets/images/yoga.png",
      description:
        "ZenFlow is a deep learning–based web application that classifies yoga poses from images and live webcam video using transfer learning with the Xception architecture.",
      goal:
        "To enhance yoga practice and instruction by offering an intuitive, real-time pose classification tool that provides feedback on alignment and posture, supporting both personal practice and research-based analysis.",
      skills: [
        "Applied transfer learning with Xception CNN for pose classification",
        "Preprocessed and augmented image datasets for robust model training",
        "Built, trained, and evaluated a deep learning model for multi-class pose prediction",
        "Developed an interactive Flask-based web application with image and webcam support",
        "Implemented real-time video processing using OpenCV",
        "Integrated frontend UI using HTML, CSS, and JavaScript for seamless user experience"
      ],
      tools: {
        languages: ["Python"],
        aiModel: "Xception (pre-trained on ImageNet)",
        libraries: ["Numpy", "TensorFlow", "Keras", "OpenCV"],
        hosting: "Flask web app",
        media: "Images and live webcam video for pose classification"
      },
      github: "https://github.com/ptrishita/ZenFlow-Enlightened-Yoga-Pose-Classification-Via-Transfer-Learning",
      demo: "https://youtu.be/cQk1olFn-aU"
    }    
    
  ];

  const featuredGallery = document.getElementById('featured-gallery');
  const otherGallery = document.getElementById('other-gallery');
  const modal = document.getElementById('project-modal');
  const modalBody = document.getElementById('modal-body');
  const closeBtn = document.getElementById('close-btn'); // <-- Using ID here

  projects.forEach((project, index) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <img src="${project.image}" alt="${project.title}" />
      <div class="overlay">
        <button data-index="${index}">Details</button>
      </div>
      <h3 class="project-title">${project.title}</h3>
    `;

    // 👇 Decide where to put the card
    if (project.category === "featured") {
      featuredGallery.appendChild(card);
    } else {
      otherGallery.appendChild(card);
    }
  });

  document.addEventListener('click', (e) => {
    if (e.target.tagName === 'BUTTON' && e.target.dataset.index !== undefined) {
      const index = parseInt(e.target.dataset.index);
      const project = projects[index];

      // Build link section correctly
      let links = `<a href="${project.github}" target="_blank">🔗 GitHub</a>`;
      if (project.demo) {
        links += ` | <a href="${project.demo}" target="_blank">🌐 Live Demo</a>`;
      }

      modalBody.innerHTML = `
        <h2>${project.title}</h2>
        <p><strong>Description:</strong> ${project.description}</p>
        <p><strong>Goal:</strong> ${project.goal}</p>
        <p><strong>Skills:</strong></p>
        <ul>${project.skills.map(skill => `<li>${skill}</li>`).join('')}</ul>
        <p><strong>Tools & Technologies:</strong></p>
        <ul>
          <li><strong>Languages:</strong> ${project.tools.languages.join(', ')}</li>
          <li><strong>AI Model:</strong> ${project.tools.aiModel}</li>
          <li><strong>Libraries:</strong> ${project.tools.libraries.join(', ')}</li>
          <li><strong>Hosting:</strong> ${project.tools.hosting}</li>
          <li><strong>Media:</strong> ${project.tools.media}</li>
        </ul>
        <p>${links}</p>
      `;
      modal.style.display = 'block';
    }
  });

  closeBtn.addEventListener('click', () => {
    modal.style.display = 'none';
  });

  window.addEventListener('click', (e) => {
    if (e.target === modal) {
      modal.style.display = 'none';
    }
  });
});


