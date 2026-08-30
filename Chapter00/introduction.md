# Pedagogy 

*The following material is currently unpublished, but it is shared
publicly in a dedicated repository[^1]. Our team (Dr. Willow Fox Fortino, Dr. Federica
Bianco, and Dr. Farid Qamar, who co-taught the course in 2024) is in
discussion with publishers (Princeton Press) to gauge their interest in
turning these teaching materials into an undergraduate level textbook.*

## Introduction 

### Motivation

Data science has become an essential skill across all scientific
disciplines
 [1-12]..
The ability to analyze data, build models, and interpret results is no
longer the exclusive purview of statisticians and computer scientists.
These skills are increasingly expected of researchers and students in
physics and astronomy, and any discipline that deals with data. Students
arrive in the classroom with diverse academic backgrounds
[9,13], widely varying
levels of mathematical preparation and often only a modicum of
programming experience or less; thus, the question asks itself: "How do
you teach data science?"

### Context

The course "Foundations of Data Science for Everyone" (hereafter FDSE)
was developed under National Science Foundation (NSF) award IIS 2123264
"Collaborative Research: HDR DSC: Delaware and Mid-Atlantic Data Science
Corps" (P.I. Bianco) to expand undergraduate data science education at 
the University of Delaware, Lincoln
University of Pennsylvania and Delaware State University. The materials
presented in this chapter represent the culmination of several years of
reflection and study. 

The course was designed for a broad audience; students from any major were welcome to attend.
The goal was not to produce a professional data scientist in four
months, but rather to equip students with the conceptual knowledge base
and practical skills necessary to engage with data in their own
disciplines and, above all else, the ability to teach themselves.

### Philosophy

The design of this course has several guiding principles, each one
rooted in our experience teaching students and being a student. The first
is that **practice is essential** to understanding. The process of
understanding an idea has several stages, by our own reckoning: ability
to recite the idea upon prompting; ability to recite the idea without
prompting; ability to teach the idea to a peer; ability to use the idea
to create new ideas; ability to teach the idea to a non-peer. Each of
these stages of understanding is effectively 'unlocked' by mastering the
prior stage. In essence: students *must* practice. There is no
substitute for practice.

The second guiding principle is that **students must learn to teach
themselves**. Collective human knowledge is an ever-changing landscape,
and a student's position in that landscape usually starts at the bottom
of a valley --- their sight obscured by hills and mountains, trees and
bushes. Our job as a teacher is to help the student out of one valley so
they might know how to escape the next one on their own. 
Effectively, this means our overarching goal will be to teach critical thinking skils.

And at last, a teacher must always ask themselves: **"What does my
student *not* know?"** The most severe barrier to effective teaching I
have seen is an instructor with an unyielding perspective. From moment
to moment or from semester to semester, a teacher must always evaluate
their instruction from the perspective of their student.

There is a wealth of pedagogical
literature that supports their value. For example, there are innumerable
studies done on the importance of practice as a part of learning
[7, 14-18],
as well as the need for students to cooperate and guide their own
learning
[19-24].
Self-determined learning, known as heutagogy, is a learning paradigm
that emphasizes the the learner's independence and agency in their
education, and is being actively researched
[@25,26]. The "curse of
knowledge" is the phenomenon of experienced teachers failing to
understand what it was once like to not be an expert
[27,28].

### The theoretical framework of Cognitive Apprenticeship Theory

Teaching Data Science is a particularly well-suited discipline for the applications of active learning [29] and Cognitive Apprenticeship Theory [30]. Since it deals with teaching methods for problem-solving, it is suited to be taught with direct involvement and intense use of practical tasks. Given that it is not domain-specific, data can be chosen to resonate with phenomena relevant to students' lives.

Here, I review the key dimensions and five core teaching methods that constitute Cognitive Apprenticeship and explain how they are integrated in the Foundation of Data Science for Everyone course.



For a cognitive apprenticeship to be effective, the learning environment must be designed around four key dimension [30]: 
- **Content: domain knowledge —facts and concepts— and the “tricks of the trade” —heuristic strategies, control strategies for managing one's own thinking, and learning strategies for acquiring new knowledge.** In Data Science classes this can be implemented by showing the process via live coding. In a typical lecture, a new method is reviewed from a high-level theoretical perspective (e.g., classifications and regression trees), first conceptually, then mathematically with step-by-step examples (e.g., “Let’s calculate the impurity of some of the nodes in Figure 5.8 of the text book. Consider the root node, where the total number of samples is $N = N_1 + N_2 = 9 + 5 = 14$. The relative frequency of 'play' is $N_1/N = 64\%$, and for 'don't play' it's $N_2/N = 36\%$. To calculate the Gini impurity:

$$\begin{aligned}
G &= 1 - (p_1^2 + p_2^2) \\
G &= 1 - \left( \left(\frac{N_1}{N}\right)^2 + \left(\frac{N_2}{N}\right)^2 \right) \\
G &= 1 - \left( 0.64^2 + 0.36^2 \right) \\
G &= 46\%.
\end{aligned}$$

Now that we have some practice calculating the Gini impurity on the root node, let's do it for each of the three nodes after the root node" (Section 5.7).) Then, in a live coding session, demonstrate the application in real time (see chapter 5). Notebooks are prepared ahead of class, of course, but "reconstructed" in front of the student line by line so that the student can see the reasoning process, see how the instructor "debugs" (their own) mistakes, and how they validate, understand and interpret the results the code is showing.

- **Sequencing: Learning activities should be sequenced to reflect the changing demands of learning: increasing in complexity, increasing in diversity, and teaching global concepts before local skills. The environment should start with simpler, more familiar tasks and gradually introduce more complex and diverse problems.** Data Science is naturally suited to this process since there is a natural sequence in the discipline: tools underpin the whole infrastructure, so coding and statistics are addressed first. As part of this, students receive a gentle introduction to data with simple datasets to demonstrate what was taught about coding and statistics. Then they use the tools to start working on data preparation. Meanwhile, they are learning about different models and can start on model selection, then model implementation. The curriculum outlined below follows this paradigm.

- **Sociology: This dimension emphasizes the social context of learning. It involves creating a culture of expert practice where students learn through social interaction, cooperation, and by becoming part of a "community of practice".** Once again, data science as a chiefly collaborative discipline where different skills are required naturally provides this. Students in FDSE are encouraged to work in groups, where individual skills are expected to be elevated: statistics, domain knowledge, coding expertise, visualization skills, and writing skills; all and more can be key to providing one's contribution. Students are actively encouraged to reflect on how they individually contribute to the work: while they can turn in assignments as a group, they are required to indicate who they worked with and how they have participated in the work.

- **Methods: This encompasses the five teaching methods described below (modeling, coaching, articulation, reflection, and exploration).** The use of these methods in FDSE is detailed below, grouped into three phases (as described in [31]):


**_Phase 1: The Core Methods_**

_Modeling: An expert (typically the teacher) performs a task while explaining their reasoning out loud. This allows students to build a conceptual model of the cognitive processes required to accomplish the task._ The lectures involve live-coding in real time by the instructor (with subsets of the code assigned as in-class exercises, see Coaching and Scaffolding). Code is written in jupyter notebooks, but the notebooks are developed in real time, instead of being pre-compiled and explained in front of the students. Each line of code is described, justified, and explained as it is being written, and alternative options can be shown.

_Coaching: The teacher observes the student as they perform a task, and provides hints, feedback, and new tasks as needed. The mentor plays an active role in guiding the student’s performance._ Lectures are deliberately interrupted to give time to students, individually or in groups, to complete sections of the notebook that form the skeleton of the lecture live-coding portion. These can be tasks that review their knowledge from previous lectures (e.g., data preparation tasks after Lesson 2), or that stretch the students’ understanding of the current material (when studying Classification and Regression Trees, “How would you implement a binary choice based on one feature of this dataset, how would you decide which feature to use based on the associated Gini impurity? Give it a try in groups!”). On Zoom, the instructors alternate visiting students’ breakout rooms. In person, the classroom should be suited to walking around between groups. The instructors and TAs observes how they approach the task, ensure effective collaboration practices are in place, and correct the course or offer extension tasks as needed. When successfully implemented, this is analogous to the Think-Pair-Share methodology (Lyman 1981) of demonstrated effectiveness in physics (Rahman & Golamgouse-Toraub 2025, Gok 2018, but also see Cooper, Schinske, & Tanner).

**_Phase 2: The Methods to Generalize Learning._**
_Articulation: Students are encouraged to articulate their knowledge, reasoning, or problem-solving processes. This can involve explaining their thought process to others or a teacher, which helps solidify their understanding._ In their assignments, students are asked to make plots and explain what the plot shows. At a high level, this is the most visible task a (data) scientist performs is to share knowledge, and knowledge sharing requires introspection and articulation. In practice, this can be implemented asking the students to reflect and write down how to interpret the numerical or graphical results they observe from their analysis. A note is in order here; with recent advances in AI, a student could not only ask AI to write their code, but also ask AI to describe the resulting figures and results. The topic of teaching in the AI era and the pedagogical implications (Beale 2025, Tabak 2026) are beyond the scope of this chapter, but we are necessarily paying attention to this issue and adapting pedagogy to it.

_Reflection: Students are prompted to compare their own problem-solving processes with those of an expert, a peer, or an internal model of expertise. This helps them identify differences and improve their own strategies._ Active coding sessions end with the instructor showing the way they have solved the problem (which is generally only one of many right ways to do it). Homework solutions are assigned for review, particularly ahead of midterm and final exams.

**_Phase 3: The Method for Independent Learning_**
_Exploration: The teacher sets general goals for the student and encourages them to formulate their own sub-goals and problems to solve. This pushes students to work independently and take on the role initially held by the mentor._ With assignments set up as compact and simplified research problems from real scholarly work, the option to explore the subject to different depths is naturally available. Tasks are often offered with structured hierarchical goals, some mandatory some optional (e.g., fit a first degree polynomial to the data — mandatory for all—; fit a higher order model of your choice —mandatory for graduate students only—; choose between the models based on Null Hypothesis Rejection test practices —mandatory for graduate students only—; choose between models based on Bayesian practices —optional for all).
In more advanced classes, the final exam is implemented as a group project, where the students begin with ideating, proposing, and then finally performing the project. This is not always possible in undergraduate classes, however, due to the fast pace and learning requirements.

---

### Course Structure 

The course is organized into ten lessons that each cover a broad machine learning or data science topic. Each lesson is a written guide, explaining each topic to the appropriate depth that lets the student imagine how they might apply the concept to an example. The lesson notes are shared below for each topic with a lecture component (all but the introduction to Python coding, which is entirely developed as a live coding session). Each lesson has a companion Jupyter notebook (written in Python) that serves as an in-class live-coding exercise to immediately demonstrate the applications of the concepts shared in the lecture. Most lessons also come with an associated homework with clear objectives and criteria for assessment. These are available at the dedicated class GitHub repository, along with two exams. An example homework and both exams are also included in full in the appendix.




Lesson 1: Python 101

:   Introductory lesson on Python, starting as an absolute beginner and
    covering arithmetic, loops, conditional expressions and functions.

Lesson 2: Data Exploration

:   Exploring data in Python with `Pandas`, data correlation,
    visualizations.

Lesson 3: Statistics

:   Frequentist probability, distributions, the Law of Large Numbers,
    the Central Limit Theorem.

Lesson 4: Null Hypothesis Rejection Testing

:   Null hypothesis framework, the z-test and ks-test.

Lesson 5: Introduction To Machine Learning

:   Data, models, objective functions, data preprocessing,
    hyperparameters and optimization.

Lesson 6: Regression and Classification

:   Linear, multiple linear and logistic regression.

Lesson 7: Tree Models

:   Decision tree methods for classification.

Lesson 8: Clustering

:   $k$-means clustering, dbscan, agglomerative clustering.

Lesson 9: Inferential Neural Networks

:   Introduction to neural networks, dense layers, convolutional neural
    networks.

Lesson 10: Generative Neural Networks

:   Autoencoders for super-resolution, an overview of LLMs.

---


## References

[1] Tony Hey, Stewart Tansley, Kristin Tolle, and Jim Gray. The Fourth Paradigm: Data-Intensive Scientific Discovery. Microsoft Research, October 2009.

[2] Emilio Porcu, Roy El Moukari, Laurent Najman, Francisco Herrera, and Horst Simon. Data Science: a Natural Ecosystem, January 2026. arXiv:2506.11010 [cs.LG].

[3] Ricardo Vinuesa, Paola Cinnella, Jean Rabault, Hossein Azizpour, Stefan Bauer, Bingni W. Brunton, Arne Elofsson, Elias Jarlebring, Hedvig Kjellström, Stefano Markidis, David Marlevi, Javier García-Martínez, and Steven L. Brunton. Decoding complexity through machine learning is redefining scientific discovery. Communications Physics, 9(1):168, May 2026.

[4] Laura Antonucci, Antonio Balzanella, Elvira Bruno, Crocetta Crocetta, Simone Di Zio, Lara Fontanella, Maurizio Sanarico, Bruno Scarpa, Rosanna Verde, and Giorgio Vittadini. Data science skills for the next generation of statisticians. Statistical Journal of the IAOS, 39(4):773–782, November 2023.

[5] Valerie Fu. AI for Science: Opportunities, Challenges, and Future Directions. Journal of Data Science, 24(1):106–124, January 2026.

[6] William R. Hersh, Robert E. Hoyt, Steven Chamberlin, Jessica S. Ancker, Aditi Gupta, and Tara B. Borlawsky-Payne. Beyond mathematics, statistics, and programming: data science, machine learning, and artificial intelligence competencies and curricula for clinicians, informaticians, science journalists, and researchers. Health Systems, 12(3):255–263, July 2023.

[7] G. Longo, M. Brescia, S. G. Djorgovski, S. Cavuoti, and C. Donalek. Data Driven Discovery in Astrophysics, November 2014. arXiv:1410.5631 [astro-ph.IM].

[8] Jina Kang, Chungsoo Na, Morgan Diederich, Hillary Swanson, and Lili Yan. Cultivating science data literacy in K–16 science education through data practices: A systematic review. Instructional Science, 54(2):50, May 2026.

[9] Zofia Bednarowska-Michael and Emma Uprichard. Bringing Interdisciplinary Data Science Education Challenges into the Classroom. Journal of Statistics and Data Science Education, 34(1):72–87, January 2026.

[10] Motahareh Zarefard and Nicola Marsden. The Essential Competencies of Data Scientists: A Framework for Hiring and Training. In Human Interface and the Management of Information: Thematic Area, HIMI 2024, Held as Part of the 26th HCI International Conference, HCII 2024, Washington, DC, USA, June 29–July 4, 2024, Proceedings, Part III, pages 397–418, Berlin, Heidelberg, June 2024. Springer-Verlag.

[11] Yixiao Dong, Deodatta Baral, and Kushmakar Baral. Are U.S. graduate curricula ready to prepare social data scientists for the AI era? Frontiers in Education, 10, January 2026.

[12] Jonas Gunklach, Mario Nadj, Sven Michalczyk, Katharina Jacob, Christoph Gröger, and Alexander Mädche. Beyond the Unicorn? Job Roles in Data Science. Business & Information Systems Engineering, July 2025.

[13] Vandana Janeja, Maria Sanchez, Yi Xuan Khoo, Claudia Vacano, and Lujie Chen. Adopting foundational data science curriculum with diverse institutional contexts. pages 576–582, 03 2024.

[14] Jessica L. Alzen, Ilana M. Trumble, Kimberly J. Cho, and Eric A. Vance. Training Interdisciplinary Data Science Collaborators: A Comparative Case Study. Journal of Statistics and Data Science Education, 32(1):73–82, January 2024.

[15] Brandon A. Dickson, Douglas G. Woolford, Boba Samuels, and Donna Kotsopoulos. Active Learning in Post-Secondary Statistics and Data Sciences Teaching: Lesson-Level Moments and Course-Level Alternative Models. Journal of Statistics and Data Science Education, 34(2):223–238, April 2026.

[16] Sarah R. Powell, Emily C. Bouck, Marah Sutherland, Ben Clarke, Tessa L. Arsenault, and Shaqwana Freeman-Green. Essential Components of Math Instruction. TEACHING Exceptional Children, 56(1):14–24, September 2023.

[17] Qiaoyi Liu, Harish Moni Prakash, and Andrew F. Heckler. Algebra and other relevant physics skills: The effectiveness of mastery practice on skills accuracy and exam grade in introductory physics. Physical Review Physics Education Research, 21(1):010121, March 2025.

[18] Brendon D. Mikula and Andrew F. Heckler. Framework and implementation for improving physics essential skills via computer-based practice: Vector math. Physical Review Physics Education Research, 13:010122, June 2017.

[19] Jacob Morrison, Clara Na, Jared Fernandez, Tim Dettmers, Emma Strubell, and Jesse Dodge. Holistically Evaluating the Environmental Impact of Creating Language Models, March 2025. arXiv:2503.05804 [cs.CY].

[20] Justine Mercado. Self-Directed Learning in STEM Teaching and Learning: A Systematic Review of Empirical Evidence. *Sciencechno: Journal of Science and Technology*, 3:53–84, March 2024.

[21] Andreia Balan and Anders Jönsson. Evaluation of a self-instructional self-regulated learning material in mathematics. Frontiers in Education, 10, March 2025.

[22] Katrina E. Forbes-McKay, Pauline A.M. Bremner, Pamela Johnston, and Carol Air. Exploring independent learning (IL) and its relationship to mindset, motivated strategies for learning and academic performance. Journal of Applied Research in Higher Education, 17(1):205–218, December 2023.

[23] Almaz Mesghina, Guanglei Hong, and Adelle Durrell. Cooperative Learning in Introductory Statistics: Assessing Students' Perceptions, Performance, and Learning in Heterogeneous and Homogeneous Groups. *Journal of Statistics and Data Science Education*, 32(4):444–456, October 2024.

[24] M. P. Silverman. Self-directed learning: A heretical experiment in teaching physics. American Journal of Physics, 63(6):495–508, June 1995.

[25] Theodore Newfield. Heutagogy in Education: Fostering Self-Directed Learning for the 21st-Century. Journal of Career and Technical Education, 40, June 2025.

[26] Raju Panta. Heutagogy: A Comprehensive Review of Self-Determined Learning in Contemporary Education. *Curres*, 17(8):e89731.

[27] Itamar Shatz. The curse of knowledge when teaching statistics. Teaching Statistics, 45(1):22–26, 2023.

[28] Jeff Froyd and Jean Layne. Faculty development strategies for overcoming the "curse of knowledge". In 2008 38th Annual Frontiers in Education Conference, pages S4D-13–S4D-16, October 2008.

[29] Charles C. Bonwell and James A. Eison. *Active learning: Creating excitement in the classroom*. 1991 ASHE-ERIC Higher Education Reports. ERIC, 1991.

[30] Allan Collins, John Seely Brown, and Susan E. Newman. Cognitive apprenticeship: Teaching the crafts of reading, writing, and mathematics. In *Knowing, Learning, and Instruction*, pages 453–494. Routledge, 2018.

[31] KNILT. Unit 4: Modeling, coaching, scaffolding, articulation, reflection, and exploration. KNILT, University at Albany, 2019. Accessed: 2026-06-21.

[32] Frank Lyman. The responsive classroom discussion: The inclusion of all students. In *Mainstreaming Digest*, pages 109–113. University of Maryland, 1981.

[33] Saif Ur Rahman and Hanna Golamgouse-Torauf. The impact of a think, write, pair, and share (TWPS) activity on pupils' confidence, critical thinking, and understanding of mathematics lesson. *Discover Education*, 4(1):489, 2025.

[34] Tolga Gok. The evaluation of conceptual learning and epistemological beliefs on physics learning by think-pair-share. *Journal of Education in Science Environment and Health*, 4(1):69–80, 2018.

[35] Katelyn M. Cooper, Jeffrey N. Schinske, and Kimberly D. Tanner. Reconsidering the share of a think-pair-share: Emerging limitations, alternatives, and opportunities for research. *CBE—Life Sciences Education*, 20(1):fe1, 2021.

[36] Russell Beale. Computer Science Education in the Age of Generative AI, July 2025. arXiv:2507.02183 [cs.CY].

[37] Ian Taback. Generative AI and the Future of Data Science Education. *Journal of Statistics and Data Science Education*, 2026 (forthcoming).


