## Minimize Cost

#### Hypothesis and Cost

![CodeCogsEqn (5)](https://user-images.githubusercontent.com/62995632/93719093-662e7c00-fbbb-11ea-9805-6a5063589840.gif)

![CodeCogsEqn](https://user-images.githubusercontent.com/62995632/93712521-5bf78800-fb91-11ea-8897-2533e31787c4.gif)


#### Simplified hypothesis

![CodeCogsEqn (6)](https://user-images.githubusercontent.com/62995632/93719105-73e40180-fbbb-11ea-945d-3491a7f98730.gif)

![CodeCogsEqn (1)](https://user-images.githubusercontent.com/62995632/93718554-27e38d80-fbb8-11ea-8bb2-b2e05e19d00b.gif)


#### Gradient descent algorithm
- Minimize cost function
- Gradient descent is used many minimization problems
- For a given cost function, cost(W,b), it will find W,b to minimize cost
- It can be applied to more general function: cost(w1,w2,...)

#### How it works
- Start with initial guesses

-> Start at 0,0(or any other value)
-> Keeping changing W and b a little bit to try and reduce cost(W,b)

- Each time you change the parameters, you select the gradient which reduces cost(W,b) the most possible
- Repeat
- Do so until you converge to a local minimum
- Has an interesting property

-> Where you start can determine which minimum you end up


#### Formal definition

![CodeCogsEqn (1)](https://user-images.githubusercontent.com/62995632/93718554-27e38d80-fbb8-11ea-8bb2-b2e05e19d00b.gif)
-> ![CodeCogsEqn (2)](https://user-images.githubusercontent.com/62995632/93718904-17ccad80-fbba-11ea-9170-d3bd28f3a5c4.gif)

![CodeCogsEqn (3)](https://user-images.githubusercontent.com/62995632/93718981-9de8f400-fbba-11ea-8c89-c2af5eaa2420.gif)
-> ![CodeCogsEqn (4)](https://user-images.githubusercontent.com/62995632/93719055-22d40d80-fbbb-11ea-8637-aab83e66f35f.gif)


#### Convex function

check cost function is convex -> then we can get the result
