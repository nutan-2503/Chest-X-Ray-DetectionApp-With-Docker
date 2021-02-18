FROM python:3.7.3-stretch
RUN mkdir /app
WORKDIR /app
ENV FLASK_APP=app.py
COPY . /app
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT [ "flask"]
CMD [ "run", "-h" , "127.0.0.1", "-p", "5000"]