# How this course is made and checked

The lessons make claims with numbers in them. This page says what stands behind a claim, so you can check any of it, and so that a new lesson can be held to the same bar.

## The order of work for one lesson

1. Write the code: one short file in `gpt/`, the least that does the job.
2. Write a test for each thing the lesson says the code does.
3. Run the lesson's script. It prints its results and saves them in `results/`, with the commit, the versions and the machine.
4. Write the page from what the script printed. A sentence is on the page only if it is a test that passes, a number in `results/`, a sum on such numbers, or a source you can open.

A page written before its run tends to say what the author hoped for. So the page comes last.

## What each page must be

- **One idea**, opened by something the last lesson could not do.
- **Six sections**, the same on each page: the idea, a sum by hand, the code, the proof, what it still cannot do, and questions to check yourself.
- **At most 1,200 words** outside its code. A page that runs over gets cut.
- **Nothing used before it is defined**, in plain words. The glossary sets each plain word beside the field's own.
- **A question to predict**, which nothing above it answers. Where the run's output answers it, it stands above the run, and its answer is the first hidden block of "Check yourself".
- **Results that did not go our way stay in.** Lessons 34 and 35 each build a speed-up that is no faster. Lesson 37 says its chip timings are not settled, and lesson 41 says its script used a looser yardstick than the lesson it cites.

## What the tests enforce

`tests/test_lessons.py` runs on each push and fails the build when:

- code quoted on a page differs from the source file by one character
- a line in a page's output block is not a line its script printed, or is out of order
- a sample on a page is not the sample in the results file
- a number on a scoreboard was printed by no script
- a link or a command on a page points at nothing
- a page has more than 1,200 words, a section missing, or a word from the list of filler

Each lesson's own tests must be able to fail. For the claims that matter we plant the bug the lesson warns about and check that a test goes red.

## The five reads

Before a batch of lessons merges, five reviewers each read it fresh, with no sight of the others' notes:

1. **Claims.** Each number and each sentence about the code is checked against `results/`, the code and the earlier lessons. A cause stated with no measurement behind it is rewritten to say only what was measured.
2. **Plain writing.** Each page is scored on two public rubrics for machine-sounding prose, [skill-deslop](https://github.com/stephenturner/skill-deslop) and [stop-slop](https://github.com/hardikpandya/stop-slop), and must reach 40 of 50 on both.
3. **A newcomer.** A reader with Python and school maths, and only the earlier lessons, answers each question from the page alone and lists each place they got lost.
4. **One piece.** A reader takes a run of lessons, the README and lesson 0 together and checks that each page opens where the last one stopped, that one word means one thing, and that nothing is said twice.
5. **Code.** A reviewer reads the change against the whole repository for wrong results, tests that cannot fail, and code that could be shorter.

Findings are fixed before the merge. A real choice between two reasonable options is written down with the option taken.

## Timings

A timing is the fastest of several tries on a machine doing nothing else, and the page names the machine. Timings move from run to run, and each page that has one says so. When two runs disagree by more than that, the page gives both.

## Rented machines

Lessons 41 and 42 run on AWS. The scripts in `aws/` follow four rules:

- Each machine is started with a time limit, after which it destroys itself and its disk.
- Each machine, disk, key and door for SSH carries one tag, so what was started can be found and removed.
- SSH is open to the address that started the machine and no other.
- A run's cost is the machine's hours, from the times AWS reports, times the on-demand price. `results/42_bill.json` has ours.

## Adding a lesson

Follow the order of work above, run `uv run pytest`, and put the page through the five reads. If the lesson changes what an earlier page says, change that page in the same pull request.
