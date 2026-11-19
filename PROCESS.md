# How this course is made and checked

The lessons make claims with numbers in them. This page says what stands behind a claim, so you can check any of it, and so that we can hold a new lesson to the same bar.

## The order of work for one lesson

1. Write the code: one short file in `gpt/`, the least that does the job.
2. Write a test for each thing the lesson says the code does.
3. Run the lesson's script. It prints its results and saves them in `results/`, with the commit, the versions and the machine.
4. Write the page from what the script printed. Each claim of fact on the page rests on a test that passes, a number in `results/`, a sum on such numbers, or a source you can open.

A page written before its run tends to say what the author hoped for, so we write the page last.

## What each page must be

- A page has one idea, opened by something the last lesson could not do.
- It has six sections, the same on each page: the idea, a sum by hand, the code, the proof, what it still cannot do, and questions to check yourself.
- It has at most 1,500 words outside its code, and most pages sit near 1,200. A sentence stays if a reader would miss it: a result, a definition, a reason, a pointer. A sentence that says again what the page has said goes, whatever the count.
- It uses nothing before defining it, in plain words. The glossary sets each plain word beside the field's own.
- It asks you to predict one result, which nothing above the question answers. Where the run's output answers it, the question stands above the run, and its answer is the first hidden block of "Check yourself".
- Results that did not go our way stay in. Lessons 34 and 35 each build a speed-up that is no faster. Lesson 37 says its chip timings are not settled, lesson 41 says its script used a looser yardstick than the lesson it cites, and lesson 45 says that one rating does not suit its seven matches.

## What the tests enforce

`tests/test_lessons.py` runs on each push to `main` and on each pull request, and fails the build when:

- code quoted on a page differs from the source file by one character
- a line in a page's output block is not a line its script printed, or is out of order
- a sample on a page is not the sample in the results file
- a number on a scoreboard was printed by no script
- a link or a command on a page points at nothing
- a page has more than 1,500 words, a section missing, or a word from the list of filler

Each lesson's own tests must be able to fail. For the claims that matter we plant the bug the lesson warns about and check that a test goes red.

## The five reads

Before a batch of lessons merges, it is read five times, each read from scratch with no sight of the others' notes:

1. For claims. Each number and each sentence about the code is checked against `results/`, the code and the earlier lessons. A cause stated with no measurement behind it is rewritten to say only what was measured.
2. For plain writing. Each page is scored on two public rubrics for machine-sounding prose, [skill-deslop](https://github.com/stephenturner/skill-deslop) and [stop-slop](https://github.com/hardikpandya/stop-slop), and must reach 40 of 50 on both.
3. As a newcomer. With Python, school maths and only the earlier lessons to go on, the read answers each question from the page alone and lists each place it got lost.
4. As one piece. The read takes a run of lessons, the README and lesson 0 together and checks that each page opens where the last one stopped, that one word means one thing, and that nothing is said twice.
5. For code. The change is read against the whole repository for wrong results, tests that cannot fail, and code that could be shorter.

We fix the findings before the merge, and write down each real choice between two reasonable options with the option taken.

## One run each

Lessons 14, 15 and 16 run each setting on several seeds, and lesson 14 says how far a difference must stand from its error before we trust it. Every other result comes from one run of its setting, the rented runs included. A small difference between two such runs may be the seed's luck, and the page cannot tell you which.

## Timings

Most timings are the fastest of several tries on a machine doing nothing else, or the page says what else ran. We time a run under load once, and each run of lesson 35 too. The page names the machine. Compare timings within one page: across pages the loop, the prompt and the day differ, and one listener gets 539, 669, 643, 490 and 775 tokens a second from the same model in lessons 31, 32, 34, 35 and 37. Timings move from run to run, and each page that has one says so. If two runs disagree by more than that, the page gives both.

## Rented machines

Lessons 41, 42, 44 and 45 run on AWS. The scripts in `aws/` follow four rules:

- Each machine is started with a time limit, after which it destroys itself and its disk.
- Each machine, disk, key and door for SSH carries one tag, so we can find and remove what we started.
- SSH is open to the address that started the machine and no other.
- A run's cost is the machine's hours, from the times AWS reports, times the on-demand price, plus its disk. `results/42_bill.json` and `results/45_bill.json` have ours, and `aws/cost.sh` lists what the machines still running have cost so far.

## Adding a lesson

Follow the order of work above, run `uv run pytest`, and put the page through the five reads. If the lesson changes what an earlier page says, change that page in the same pull request.
