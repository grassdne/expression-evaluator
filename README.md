# expression-evaluator
No eval.
Lexes string, performs static error analysis, converts to tree expression, pretty prints, parses into result, inverts expression tree, parses inverted tree, converts tree to string expression.

```py
====== CALCULATE ======
> -1 + |20 * -7 | % 40 / (1/2)^2 -10  

BinaryExpr(left=BinaryExpr(left=UnaryExpr(oper=UnOper.NEGATE, expr=1), oper=BiOper.PLUS, right=BinaryExpr(left=BinaryExpr(left=WrapExpr(oper=<enum 'Absolut'>, expr=BinaryExpr(left=20, oper=BiOper.TIMES, right=UnaryExpr(oper=UnOper.NEGATE, expr=7))), oper=BiOper.MODULO, right=40), oper=BiOper.DIVIDE, right=BinaryExpr(left=BinaryExpr(left=1, oper=BiOper.DIVIDE, right=2), oper=BiOper.POWER, right=2))), oper=BiOper.MINUS, right=10)

       (-)  1
    +
                      20
                |*|
                     (-)  7
            %
                40
        /
                    1
                /
                    2
            ^
                2
-
    10

RESULT: 69.0

===== Inverting Tree =====

    10
-
                2
            ^
                    2
                /
                    1
        /
                40
            %
                     (-)  7
                |*|
                      20
    +
       (-)  1

RESULT: 10.9
CONVERTED BACK: 10 - 2 ^ 2 / 1 / 40 % - 7 * 20 + - 1
```

```py
====== CALCULATE ======
> 1+2+3+4+5+6+7+8+9+10      

BinaryExpr(left=BinaryExpr(left=BinaryExpr(left=BinaryExpr(left=BinaryExpr(left=BinaryExpr(left=BinaryExpr(left=BinaryExpr(left=BinaryExpr(left=1, oper=BiOper.PLUS, right=2), oper=BiOper.PLUS, right=3), oper=BiOper.PLUS, right=4), oper=BiOper.PLUS, right=5), oper=BiOper.PLUS, right=6), oper=BiOper.PLUS, right=7), oper=BiOper.PLUS, right=8), oper=BiOper.PLUS, right=9), oper=BiOper.PLUS, right=10)

                                    1
                                +
                                    2
                            +
                                3
                        +
                            4
                    +
                        5
                +
                    6
            +
                7
        +
            8
    +
        9
+
    10

RESULT: 55

===== Inverting Tree =====

    10
+
        9
    +
            8
        +
                7
            +
                    6
                +
                        5
                    +
                            4
                        +
                                3
                            +
                                    2
                                +
                                    1

RESULT: 55
CONVERTED BACK: 10 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1
```
