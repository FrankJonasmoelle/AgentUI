LangGraph Agent with UI
=======================

A simple implementation of a **LangGraph agent** with a **Streamlit chat interface**.

Prerequisites
-------------

- Python 3.9+
- An OpenAI API key

Installation
------------

Install the required dependencies:

.. code-block:: bash

   pip install -r requirements.txt


Configuration
-------------

Create a `.env` file in the project root and add your OpenRouter / OpenAI API key:

.. code-block:: bash

   OPENAI_API_KEY=your-api-key
   OPENROUTER_API_KEY=your-api-key


Run the Application
-------------------

Create the vector embeddings:

.. code-block:: bash

   python3 ingest.py


Run the backend: 

.. code-block:: bash

   python3 server.py


And the frontend:

.. code-block:: bash

   cd frontend/
   npm run dev


Alternatively run with Streamlit:

.. code-block:: bash

   streamlit run app.py



Have fun with your personal agent!