cd ..

python3 main.py -m gemini -d 2019 -p zeroshot
python3 main.py -m gemini -d 2017 -p zeroshot

python3 main.py -m gemini -d 2019 -p fewshot -e 1
python3 main.py -m gemini -d 2017 -p fewshot -e 1

python3 main.py -m gemini -d 2019 -p fewshot -e 2
python3 main.py -m gemini -d 2017 -p fewshot -e 2

python3 main.py -m gemini -d 2019 -p fewshot -e 3
python3 main.py -m gemini -d 2017 -p fewshot -e 3