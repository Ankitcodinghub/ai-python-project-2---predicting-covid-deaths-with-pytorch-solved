# ai-python-project-2---predicting-covid-deaths-with-pytorch-solved
**TO GET THIS SOLUTION VISIT:** [AI-python Project 2 – Predicting COVID Deaths with PyTorch Solved](https://www.ankitcodinghub.com/product/python-p2-6-of-grade-predicting-covid-deaths-with-pytorch-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;119420&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;AI-python  Project 2 - Predicting COVID Deaths with PyTorch Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
Overview

Learning objectives: * multiply tensors * check whether GPUs are available * optimize inputs to minimize outputs * use optimization to optimize regression coefficients

Note that if you normally use VS Code with Jupyter, your setup for this project will be tricky because you can’t SSH into a container. There are some notes here (https://github.com/cs544-wisc/f23/tree/main/docs/vs-code-jupyter) about how to connect — ignore if you’re using Jupyter through your browser.

Before starting, please review the general project directions.

Corrections/Clarifications

Sep 21: updated note about data source

Sep 21: add note about VS code

Sep 25: note that for Jupyter cell answers, you need to put the Python expression computing the answer on the list line of the cell (prints are for your own debugging info and do not count as cell output)

Part 1: Setup

Build the Dockerfile we give you (feel to make edits if you like) to create your environment. Run the container, setup an SSH tunnel, and open JupyterLab in your browser. Create a notebook called p2.ipynb in the nb directory.

Use train.csv and test.csv to construct four PyTorch float64 tensors: trainX, trainY, testX, and testY. Hints:

you can use pd.read_csv to load CSVs to DataFrames you can use use df.values to get a numpy array from a DataFrame you can convert from numpy to PyTorch (https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) you’ll need to do some 2D slicing: the last column contains your y values and the others contain your X values trainX (number of positive COVID tests per tract, by age group) should look like this:

python tensor([[ 24., 51., 44., …, 61., 27., 0.], [ 22., 31., 214., …, 9., 0., 0.], [ 84., 126., 239., …, 74., 24., 8.], …, [268., 358., 277., …, 107., 47., 7.], [ 81., 116., 90., …, 36., 9., 0.], [118.,

156., 197., …, 19., 0., 0.]], dtype=torch.float64)

trainY (number of COVID deaths per tract) should look like this (make sure it is vertical, not 1 dimensional!):

python tensor([[3.], [2.], [9.], …, [5.], [2.], [5.]], dtype=torch.float64)

Q1: about how many bytes does trainX consume?

Don’t count any overhead for the Python object — just multiply the element size by the number of elements.

Q2: what is the biggest difference we would have any one cell if we used float16 instead of float64?

Convert trainX to float16, then back again to float64. Subtract the resulting matrix from the original. Find the biggest absolute difference, and display as a Python float.

Q3: is a CUDA GPU available on your VM?

Write a code snippet to produce a True/False answer.

Part 2: Prediction with Hardcoded Model

Let’s predict the number of COVID deaths in the test dataset under the assumption that the deathrate is 0.004 for those &lt;60 and 0.03 for those &gt;=60. Encode these assumptions as coefficients in a tensor by pasting the following:

python coef = torch.tensor([ [0.0040], [0.0040], [0.0040], [0.0040], [0.0040], [0.0040], # POS_50_59_CP

[0.0300], # POS_60_69_CP [0.0300], [0.0300], [0.0300] ], dtype=trainX.dtype) coef

Q4: what is the predicted number of deaths for the first census tract?

Multiply the first row testX by the coef vector and use .item() to print the predicted number of deaths in this tract.

Q5: what is the average number of predicted deaths, over the whole testX dataset?

Part 3: Optimization

Let’s say y = x^2 – 8x + 19. We want to find the x value that minimizes y.

Q6: first, what is y when x is a tensor containing 0.0? python x = torch.tensor(0.0) y = ???? float(y) Q7: what x value minimizes y?

Write an optimization loop that uses torch.optim.SGD. You can experiment with the training rate and number of iterations, as long as you find a setup that gets approximately the right answer.

Part 4: Linear Regression

Use the torch.zeros function to initialize a 2-dimensional coef matrix of size and type that allows us to compute trainX @ coef (we won’t bother with a bias factor in this exercise).

Q8: what is the MSE (mean-square error) when we make predictions using this vector of zero coefficients?

You’ll be comparing trainX @ coef to trainY Optimization

In part 1, you used a hardcoded coef vector to predict COVID deaths. Now, you will start with zero coefficients and optimize them.

Seed torch random number generation with 544.

Setup a training dataset and data loader like this:

python ds = torch.utils.data.TensorDataset(trainX, trainY) dl = torch.utils.data.DataLoader(ds, batch_size=50, shuffle=True)

Write a training loop to improve the coefficients. Requirements: * use the torch.optim.SGD optimizer * use 0.000002 learning rate * run for 500 epochs * use torch.nn.MSELoss for your loss function

Q9: what is the MSE over the training data, using the coefficients resulting from the above training?

Q10: what is the MSE over the test data?

Submission

You should commit your work in a notebook named p2.ipynb.

Tester

After copying ../tester.py, ../nbutils.py, and autograde.py to your repository (where your nb/ directory is located), you can check your notebook answers with this command:

sh python3 autograde.py For the autograder to work, for each question, please include a line of comment at the beginning of code cell that outputs the answer. For example, the code cell for question 7 should look like “`python q7

… “`
