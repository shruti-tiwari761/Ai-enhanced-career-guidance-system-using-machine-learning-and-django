from django.shortcuts import render,HttpResponse,redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.core.exceptions import ValidationError
from django.contrib.auth.models import User
from django.db.utils import IntegrityError
from django.contrib.auth import authenticate, login as auth_login
import numpy as np
import joblib
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import re
from django.contrib.auth.decorators import login_required
from .models import Profile


model=joblib.load('static\Student_Career_Predicator')
model1=joblib.load('static\ew910_Predicator')
model2=joblib.load('static\Arts_Predicator')
model3=joblib.load('static\Bio_Predicator')
model4=joblib.load('static\Commerce_Predicator')
model5=joblib.load('static\Maths_Final_Predicator')
model6=OllamaLLM(model="llama3.2")
template = """
You are a Career Guidance Bot designed to assist students in exploring career paths. Provide personalized recommendations based on the user's skills and interests. 
Your responses should be:
1. Well-structured with clear paragraphs
2. Between 100-200 words
3. Include bullet points or numbered lists where appropriate
4. Focus on career-related guidance only

If the user asks something unrelated to career guidance, respond with: 
"I'm sorry, I can only assist with career-related guidance."

Here is the conversation history:
{context}

User's Question: {question}

Your Response:
"""
prompt = ChatPromptTemplate.from_template(template)

# Predefined greetings
greetings = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi! How can I help with your career queries?",
    "good morning": "Good morning! What career-related questions can I help with?",
    "good afternoon": "Good afternoon! Let me know how I can assist.",
    "good evening": "Good evening! Feel free to ask your questions.",
    "good night": "Good night! Let me know if there's something I can help with before you rest."
}

# Formatting function
def format_response(response):
    """
    Formats the chatbot response to be well-structured and within word limits.
    - Ensures response is between 100-200 words
    - Formats headings, subheadings, and bold text
    - Formats paragraphs with proper spacing
    - Converts lists into bullet points
    """
    # Split response into words
    words = response.split()
    
    # Ensure response is between 100-200 words
    if len(words) < 100:
        response = " ".join(words + ["Please provide more details about your career interests and goals so I can give you a more comprehensive response."])
    elif len(words) > 200:
        response = " ".join(words[:200]) + "..."

    # Format main headings
    response = re.sub(r'^#\s*(.*?)$', r'<h2>\1</h2>', response, flags=re.MULTILINE)
    
    # Format subheadings
    response = re.sub(r'^##\s*(.*?)$', r'<h3>\1</h3>', response, flags=re.MULTILINE)
    
    # Format bold text
    response = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', response)
    
    # Format bullet points (• or *)
    response = re.sub(r'^[•*]\s+(.*?)$', r'<li>\1</li>', response, flags=re.MULTILINE)
    
    # Format numbered lists
    response = re.sub(r'^\d+\.\s+(.*?)$', r'<li>\1</li>', response, flags=re.MULTILINE)
    
    # Format step-by-step sections
    response = re.sub(r'^Step\s+\d+:\s+(.*?)$', r'<h3>\1</h3>', response, flags=re.MULTILINE)
    
    # Format paragraphs
    paragraphs = response.split('\n\n')
    formatted_paragraphs = []
    for para in paragraphs:
        if not para.startswith(('<h2>', '<h3>', '<ul>', '<li>')):
            formatted_paragraphs.append(f'<p>{para}</p>')
        else:
            formatted_paragraphs.append(para)
    
    response = '\n'.join(formatted_paragraphs)
    
    # Wrap lists in proper HTML tags
    response = re.sub(r'(<li>.*?</li>\n?)+', r'<ul>\g<0></ul>', response)
    
    return response

# Chat view
def chat(request):
    if request.method == "POST":
        user_input = request.POST.get("user_input")
        context = request.session.get("context", "")

        # Check if input is a greeting
        if user_input.lower() in greetings:
            response = greetings[user_input.lower()]
        else:
            try:
                formatted_prompt = f"{context}\nUser: {user_input}\nAI:"
                raw_response = model6.invoke(formatted_prompt)
                response = format_response(raw_response)
            except Exception as e:
                response = "I'm sorry, something went wrong. Please try again later."

        # Update the context
        context += f"User: {user_input}\nAI: {response}\n"
        request.session["context"] = context

        # Pass structured messages for the template
        messages = [{"sender": "User", "text": user_input}, {"sender": "AI", "text": response}]
        return render(request, "chat.html", {"messages": messages})

    # On GET, show an empty chat page
    context = request.session.get("context", "")
    messages = []
    if context:
        for line in context.splitlines():
            if line.startswith("User:"):
                messages.append({"sender": "User", "text": line[6:]})
            elif line.startswith("AI:"):
                messages.append({"sender": "AI", "text": line[4:]})
    return render(request, "chat.html", {"messages": messages})
pred_mapping={
    0: "AI ML Specialist",
    1: "API Specialist",
    2: "Application Support Engineer",
    3: "Business Analyst",
    4: "Customer Service Executive",
    5: "Cyber Security Specialist",
    6: "Data Scientist",
    7: "Database Administrator",
    8: "Graphics Designer",
    9: "Hardware Engineer",
    10: "Helpdesk Engineer",
    11: "Information Security Specialist",
    12: "Networking Engineer",
    13: "Project Manager",
    14: "Software Developer",
    15: "Software Tester",
    16: "Technical Writer"
}

pred_mathematics_mapping={
    0: 'Pure Mathematics, Applied Mathematics, Data Science, Machine Learning, Actuarial Science, or Mathematical Modelling',
    1: 'Statistics, Mathematical Biology, Econometrics, or Operations Research, or Bioinformatics',
    2: 'Consider exploring fields outside of mathematics, such as Humanities or Business',
}

pred_bio_mapping={
    0: 'Bachelor of Veterinary Science, Environmental Science, Biomedical Engineering, or Molecular Biology',
    1: 'Botany, Zoology, Forensic Science, or Agricultural Science',
    2: 'Explore other fields outside Biology',
    3: 'MBBS, Bachelor of Science in Biology, Biotechnology, Microbiology, or Biochemistry'
}

pred_com_mapping={
    0: 'Bachelor of Business Administration (BBA), Bachelor of Commerce (B.Com), Chartered Accountancy (CA), Finance and Economics, or International Business',
    1: 'Economics, Marketing, Business Studies, Supply Chain Management, or Human Resource Management',
    2: 'Explore other fields outside Commerce, such as Arts, Humanities, or Technology'

}

pred_arts_mapping={
    0: 'Consider exploring alternative fields, such as STEM or Commerce',
    1: 'Fine Arts, Psychology, Journalism, Creative Writing, or Literature',
    2: 'History, Sociology, Political Science, Cultural Studies, or Philosophy'

}

pred_9_10_mapping={
    0: 'Arts',
    1: 'Commerce',
    2: 'Science (Biology)',
    3: 'Science (Mathematics)', 
    4: 'Undecided'

}


# Create your views here.
def index(request):
    # return HttpResponse("This is the home page")
    return render(request,'index.html')



def about(request):
    # return HttpResponse("This is the home page")
    return render(request,'about.html')

def contact(request):
    # return HttpResponse("This is the home page")
    return render(request,'contact.html')
def classpage(request):
    return render(request,'class.html')
def class9thand10th(request):
    output = None
    if request.method == 'POST':
        try:
            # Collecting form inputs from the POST request
            grades_mathematics = request.POST.get('grades-mathematics', 'C')
            grades_science = request.POST.get('grades-science', 'C')
            grades_english = request.POST.get('grades-english', 'C')
            grades_social_studies = request.POST.get('grades-social-studies', 'C')
            grades_computer_science = request.POST.get('grades-computer-science', 'C')
            
            interest_mathematics = request.POST.get('interest-mathematics', 'Moderate Interest')
            interest_science = request.POST.get('interest-science', 'Moderate Interest')
            interest_english = request.POST.get('interest-english', 'Moderate Interest')
            interest_social_studies = request.POST.get('interest-social-studies', 'Moderate Interest')
            interest_computer_science = request.POST.get('interest-computer-science', 'Moderate Interest')
            
            preferred_career_options = request.POST.get('preferred-career-options', 'No Specific Preference')
            preferred_stream_11th_grade = request.POST.get('preferred-stream-11th-grade', 'Undecided')
            learning_style = request.POST.get('learning-style', 'Visual')
            interest_new_concepts = request.POST.get('interest-new-concepts', 'Moderate Interest')

            # Ensure you have 16 features in the input data as expected by the model
            value_mapping = {
                # Grades mapping
                "A": 0,
                "B": 1,
                "C": 2,
                "D": 3,
                "E": 4,
                
                # Interest mapping
                "High Interest": 0,
                "Moderate Interest": 1,
                "Low Interest": 2,
                "No Interest": 3,
                
                # Career options mapping
                "Architecture": 0,
                "Arts/Graphics Designing": 1,
                "Business": 2,
                "Computer Science": 3,
                "Engineering": 4,
                "Environmental Science": 5,
                "Law": 6,
                "Medicine": 7,
                "No Specific Preference": 8,
                "Psychology": 9,
                
                # Preferred stream for 11th grade
                "Arts": 0,
                "Commerce": 1,
                "Science(Biology)": 2,
                "Science(Mathematics)": 3,
                "Undecided": 4,
                
                # Learning style mapping
                "Auditory": 0,
                "Kinesthetic": 1,
                "Reading/Writing": 2,
                "Visual": 3,
                
                # Interest in learning new concepts/technology mapping
                "High Interest": 0,
                "Moderate Interest": 1,
                "Low Interest": 2,
                "No Interest": 3,
            }

            # Convert inputs to numerical values, ensuring 16 features are included
            input_data = np.array([[
                value_mapping[grades_mathematics], value_mapping[grades_science],
                value_mapping[grades_english], value_mapping[grades_social_studies],
                value_mapping[grades_computer_science], value_mapping[interest_mathematics],
                value_mapping[interest_science], value_mapping[interest_english],
                value_mapping[interest_social_studies], value_mapping[interest_computer_science],
                value_mapping[preferred_career_options], value_mapping[preferred_stream_11th_grade],
                value_mapping[learning_style], value_mapping[interest_new_concepts]
              # Additional features to match the model's requirement (adjust as necessary)
            ]])

            # Replace 'model1' with your actual trained model
            pred = model1.predict(input_data)  # Example prediction step

            # You can customize this with actual prediction decoding logic if necessary
            decoded_pred = pred_9_10_mapping.get(pred[0], "Unknown")

            output = f"Predicted Career Path: {decoded_pred}"

        except KeyError as e:
            output = f"Error: Missing or invalid input - {str(e)}"
        except Exception as e:
            output = f"An unexpected error occurred: {str(e)}"

    return render(request, 'class9thand10th.html', {'output': output})
def Math(request):
    #return render(request,'Mathsprediction.html')
    output = None
    if request.method == 'POST':
        try:
            # Collecting form inputs from the POST request
            math_confidence = int(request.POST.get('math-confidence', 0))
            math_performance = int(request.POST.get('math-performance', 0))
            career_interest_math = int(request.POST.get('career-interest-math', 0))
            career_importance_math = int(request.POST.get('career-importance-math', 0))
            problem_solving_skills_math = int(request.POST.get('problem-solving-skills-math', 0))
            math_projects = int(request.POST.get('math-projects', 0))
            teamwork_preference_math = int(request.POST.get('teamwork-preference-math', 0))
            tech_comfort_math = int(request.POST.get('tech-comfort-math', 0))

            # Store all inputs into a NumPy array
            input_data = np.array([[math_confidence, math_performance, career_interest_math, 
                                    career_importance_math, problem_solving_skills_math, 
                                    math_projects, teamwork_preference_math, tech_comfort_math]])

            # Example: Predict using the model
            pred = model5.predict(input_data)  # Replace `model` with your trained model instance
            decoded_pred = pred_mathematics_mapping.get(pred[0], "Unknown")  # Replace `pred_mapping` with your decoding logic

            output = f"{decoded_pred}"
        except ValueError as e:
            output = f"Error: {str(e)}"
        except Exception as e:
            output = f"An unexpected error occurred: {str(e)}"

    return render(request, 'Mathsprediction.html', {'output': output})
def Science(request):
    #return render(request,'Scienceprediction.html')
    output = None
    if request.method == 'POST':
        try:
            # Collecting form inputs from the POST request
            biology_confidence = request.POST.get('biology-confidence', "Neutral/Somewhat Confident")
            biology_performance = request.POST.get('biology-performance', "Average")
            career_interest_biology = request.POST.get('career-interest-biology', "Neutral/Somewhat Interested")
            career_importance_biology = request.POST.get('career-importance-biology', "Somewhat Important")
            problem_solving_skills_biology = request.POST.get('problem-solving-skills-biology', "Average")
            science_projects = request.POST.get('science-projects', "Sometimes")
            teamwork_preference_biology = request.POST.get('teamwork-preference-biology', "It depends")
            lab_work_comfort = request.POST.get('lab-work-comfort', "Neutral/Somewhat Comfortable")

            # Map categorical values to numerical values (example mapping logic)
            value_mapping = {
                "Confident": 0,
                "Neutral/Somewhat Confident": 1,
                "Not Confident": 2,
                "Slightly Confident": 3,
                "Very Confident": 4,
                "Average": 0,
                "Below Average": 1,
                "Excellent": 2,
                "Good": 3,
                "Poor": 4,
                "Interested": 0,
                "Neutral/Somewhat Interested": 1,
                "Not Interested": 2,
                "Slightly Interested": 3,
                "Very Interested": 4,
                "Not Important": 0,
                "Somewhat Important": 1,
                "Very Important": 2,
                "No": 0,
                "Sometimes": 1,
                "Yes": 2,
                "It depends": 0,
                "Comfortable": 0,
                "Neutral/Somewhat Comfortable": 1,
                "Not Comfortable": 2,
                "Slightly Comfortable": 3,
                "Very Comfortable": 4,
            }

            # Convert inputs to numerical values
            input_data = np.array([[value_mapping[biology_confidence], value_mapping[biology_performance],
                                    value_mapping[career_interest_biology], value_mapping[career_importance_biology],
                                    value_mapping[problem_solving_skills_biology], value_mapping[science_projects],
                                    value_mapping[teamwork_preference_biology], value_mapping[lab_work_comfort]]])

            # Predict using the model
            pred = model3.predict(input_data)  # Replace `model` with your trained model instance
            decoded_pred = pred_bio_mapping.get(pred[0], "Unknown")  # Replace `pred_mapping` with your decoding logic

            output = f"{decoded_pred}"
        except KeyError as e:
            output = f"Error: Missing or invalid input - {str(e)}"
        except Exception as e:
            output = f"An unexpected error occurred: {str(e)}"

    return render(request, 'Scienceprediction.html', {'output': output})
def Commerce(request):
    #return render(request,'Commerceprediction.html')
    output = None
    if request.method == 'POST':
        try:
            # Collecting form inputs from the POST request
            commerce_confidence = request.POST.get('commerce-confidence', "Neutral/Somewhat Confident")
            commerce_performance = request.POST.get('commerce-performance', "Average")
            career_interest_commerce = request.POST.get('career-interest-commerce', "Neutral/Somewhat Interested")
            career_importance_commerce = request.POST.get('career-importance-commerce', "Somewhat Important")
            problem_solving_skills = request.POST.get('problem-solving-skills', "Average")
            business_projects = request.POST.get('business-projects', "Sometimes")
            teamwork_preference = request.POST.get('teamwork-preference', "It depends")
            technology_comfort = request.POST.get('technology-comfort', "Neutral/Somewhat Comfortable")

            # Map categorical values to numerical values (adjust mapping if necessary)
            value_mapping = {
                "Confident": 0,
                "Neutral/Somewhat Confident": 1,
                "Not Confident": 2,
                "Slightly Confident": 3,
                "Very Confident": 4,
                "Average": 0,
                "Below Average": 1,
                "Excellent": 2,
                "Good": 3,
                "Poor": 4,
                "Interested": 0,
                "Neutral/Somewhat Interested": 1,
                "Not Interested": 2,
                "Slightly Interested": 3,
                "Very Interested": 4,
                "Not Important": 0,
                "Somewhat Important": 1,
                "Very Important": 2,
                "No": 0,
                "Sometimes": 1,
                "Yes": 2,
                "It depends": 0,
                "Comfortable": 0,
                "Neutral/Somewhat Comfortable": 1,
                "Not Comfortable": 2,
                "Slightly Comfortable": 3,
                "Very Comfortable": 4,
            }

            # Convert inputs to numerical values
            input_data = np.array([[value_mapping[commerce_confidence], value_mapping[commerce_performance],
                                    value_mapping[career_interest_commerce], value_mapping[career_importance_commerce],
                                    value_mapping[problem_solving_skills], value_mapping[business_projects],
                                    value_mapping[teamwork_preference], value_mapping[technology_comfort]]])

            # Predict using the model
            pred = model4.predict(input_data)  # Replace `model` with your trained model instance
            decoded_pred = pred_com_mapping.get(pred[0], "Unknown")  # Replace `pred_mapping` with your decoding logic

            output = f"{decoded_pred}"
        except KeyError as e:
            output = f"Error: Missing or invalid input - {str(e)}"
        except Exception as e:
            output = f"An unexpected error occurred: {str(e)}"

    return render(request, 'Commerceprediction.html', {'output': output})
def Arts(request):
    #return render(request,'Artsprediction.html')
    output = None
    if request.method == 'POST':
        try:
            # Collecting form inputs from the POST request
            arts_confidence = request.POST.get('arts-confidence', "Neutral/Somewhat Confident")
            arts_performance = request.POST.get('arts-performance', "Average")
            career_interest = request.POST.get('career-interest', "Neutral/Somewhat Interested")
            career_importance = request.POST.get('career-importance', "Somewhat Important")
            creative_thinking = request.POST.get('creative-thinking', "Neutral/Somewhat Creative")
            creative_projects = request.POST.get('creative-projects', "Sometimes")
            independent_collaborative = request.POST.get('independent-collaborative', "It depends on the project")
            public_expression = request.POST.get('public-expression', "Neutral/Somewhat Comfortable")

            # Map categorical values to numerical values (adjust mapping if necessary)
            value_mapping = {
    "Confident": 0,
    "Neutral/Somewhat Confident": 1,
    "Not Confident": 2,
    "Slightly Confident": 3,
    "Very Confident": 4,
    
    "Average": 0,
    "Below Average": 1,
    "Excellent": 2,
    "Good": 3,
    "Poor": 4,
    
    "Interested": 0,
    "Neutral/Somewhat Interested": 1,
    "Not Interested": 2,
    "Slightly Interested": 3,
    "Very Interested": 4,
    
    "Not Important": 0,
    "Somewhat Important": 1,
    "Very Important": 2,
    
    "No": 0,
    "Sometimes": 1,
    "Yes": 2,
    
    "I prefer working independently": 0,
    "I prefer working with others": 1,
    "It depends on the project": 2,
    
    "Comfortable": 0,
    "Neutral/Somewhat Comfortable": 1,
    "Not Comfortable": 2,
    "Slightly Comfortable": 3,
    "Very Comfortable": 4,
    
    "Creative": 0,  # Added missing value for creative thinking
    "Neutral/Somewhat Creative": 1,  # Added missing value
    "Not Creative": 2,
    "Slightly Creative": 3,
    "Very Creative": 4,
}
            # Convert inputs to numerical values
            input_data = np.array([[value_mapping[arts_confidence], value_mapping[arts_performance],
                                    value_mapping[career_interest], value_mapping[career_importance],
                                    value_mapping[creative_thinking], value_mapping[creative_projects],
                                    value_mapping[independent_collaborative], value_mapping[public_expression]]])

            # Predict using the model
            pred = model2.predict(input_data)  # Replace `model` with your trained model instance
            decoded_pred = pred_arts_mapping.get(pred[0], "Unknown")  # Replace `pred_mapping` with your decoding logic

            output = f"{decoded_pred}"
        except KeyError as e:
            output = f"Error: Missing or invalid input - {str(e)}"
        except Exception as e:
            output = f"An unexpected error occurred: {str(e)}"

    return render(request, 'Artsprediction.html', {'output': output})




def loginpage(request):
    # return HttpResponse("This is the home page")
    if request.method == 'POST':
        username = request.POST.get('username')  # Corrected line
        password = request.POST.get('password')
        user = authenticate(request,username=username,password=password)

        if user is not None:
            auth_login(request, user)  # Use Django's login function
            return redirect('home')  # Redirect to home page or any desired page
        else:
            # Handle invalid login
            return render(request, 'login.html', {'error': 'Invalid credentials'})
    
    else:
        return render(request,'login.html')

        
    
def signuppage(request):
    if request.method == 'POST':
        firstname = request.POST.get('firstname')
        lastname = request.POST.get('lastname')
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirmpassword = request.POST.get('confirmpassword')

        # Check if passwords match
        if password != confirmpassword:
            messages.error(request, "Passwords do not match.")
            return render(request, 'signup.html')

        # Check if username is already taken
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username is already taken.")
            return render(request, 'signup.html')

        # Check if username is alphanumeric and not purely numeric
        if not username.isalnum() or username.isnumeric():
            messages.error(request, "Username must contain both letters and numbers, and it can't be purely numeric.")
            return render(request, 'signup.html')

        # Create the user and set additional fields
        try:
            user = User.objects.create_user(username=username, email=email, password=password)
            user.first_name = firstname  # Set first name
            user.last_name = lastname    # Set last name
            user.save()

            messages.success(request, "Account created successfully.")
            return redirect('login')  # Redirect to login page after successful signup
        except ValidationError as e:
            messages.error(request, str(e))
            return render(request, 'signup.html')

    return render(request, 'signup.html')




def prediction(request):
    # return HttpResponse("This is the home page")
    output = None
    if request.method == 'POST':
        try:
            database_fundamentals = int(request.POST.get('database-fundamentals', 0))
            computer_architecture = int(request.POST.get('computer-architecture', 0))
            distributed_computing = int(request.POST.get('distributed-computing', 0))
            cyber_security = int(request.POST.get('cyber-security', 0))
            networking = int(request.POST.get('networking', 0))
            software_development = int(request.POST.get('software-development', 0))
            programming_skills = int(request.POST.get('programming-skills', 0))
            project_management = int(request.POST.get('project-management', 0))
            computer_forensics = int(request.POST.get('computer-forensics', 0))
            technical_communication = int(request.POST.get('technical-communication', 0))
            ai_ml = int(request.POST.get('ai-ml', 0))
            software_engineering = int(request.POST.get('software-engineering', 0))
            business_analysis = int(request.POST.get('business-analysis', 0))
            communication_skills = int(request.POST.get('communication-skills', 0))
            data_science = int(request.POST.get('data-science', 0))
            troubleshooting_skills = int(request.POST.get('troubleshooting-skills', 0))
            graphics_designing = int(request.POST.get('graphics-designing', 0))

        # Store all inputs into a NumPy array
            input_data = np.array([[database_fundamentals, computer_architecture, distributed_computing, 
                                    cyber_security, networking, software_development, programming_skills, 
                                    project_management, computer_forensics, technical_communication, ai_ml, 
                                    software_engineering, business_analysis, communication_skills, data_science, 
                                    troubleshooting_skills, graphics_designing]])

            # Example: Print or process input data
            pred = model.predict(input_data)
            decoded_pred = pred_mapping.get(pred[0],"Unkown")
            output = f" {decoded_pred}"
        except ValueError as e:
            output = f"Error: {str(e)}"
    return render(request,'prediction.html',{'output':output})

@login_required
def profile(request):
    try:
        profile = request.user.profile
    except Profile.DoesNotExist:
        profile = Profile.objects.create(user=request.user)

    if request.method == 'POST':
        if 'photo' in request.FILES:
            profile.photo = request.FILES['photo']
            profile.save()
            messages.success(request, 'Profile picture updated successfully!')
            return redirect('profile')

    return render(request, 'profile.html', {'profile': profile})


