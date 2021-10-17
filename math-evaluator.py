from dataclasses import dataclass
from enum import Enum, auto
from typing import Match, Set, Tuple, Union, Dict, Callable, List, TypeVar
from sys import stderr
import operator
from numbers import Number


class _Operator:

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__) + 1
        return obj

    def __init__(self, char: str, priority: int, func: Callable):
        self.char = char
        self.priority = priority
        self.func = func



class BiOper(_Operator, Enum):
    PLUS    =  '+',  4,  operator.add
    MINUS   =  '-',  4,  operator.sub
    TIMES   =  '*',  3,  operator.mul
    DIVIDE  =  '/',  3,  operator.truediv
    MODULO  =  '%',  3,  operator.mod
    POWER   =  '^',  1,  operator.pow

    def __repr__(self):
        return self.__str__()


class UnOper(_Operator, Enum):
    NEGATE   =  '-',  2,  operator.neg

    def __repr__(self):
        return self.__str__()

class _ExpressionOper:

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__) + 1
        return obj

    def __init__(self, char: str, priority=0):
        self.char = char
        self.priority = priority


class Parenth(_ExpressionOper, Enum):
    OPEN = '('
    CLOSE = ')'

    def __repr__(self):
        return self.__str__()

    

class Absolut(_ExpressionOper, Enum):
    OPEN = '|'
    CLOSE = '|'

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def func(x):
        return x if x > 0 else -x


COL_SPACING = 3
def print_expr_tree(expr, start='', unary='', wrap = ['', '']):
    if isinstance(expr, Number):
        if unary:
            print(f'{start}{wrap[0]}{unary}{wrap[0]}{" " * (COL_SPACING - 1)}{expr}')
        else:
            print(start + wrap[0] + str(expr) + wrap[1])

    elif isinstance(expr, UnaryExpr):
        print_expr_tree(expr.expr, start=start[:-1], unary=f'{wrap[0]}{unary}({expr.oper.char}){wrap[1]}')

    elif isinstance(expr, WrapExpr):
        # unary = '(-)'     wrap = ['|', '|']
        print_expr_tree(expr.expr, start, unary, [wrap[0]+expr.oper.OPEN.char, wrap[1]+expr.oper.CLOSE.char])

    else: # if isinstance(expr, BinaryExpr):
        if unary:
            s = f'{start}{unary} {wrap[0]}{expr.oper.char}{wrap[1]}'
        else:
            s = start + wrap[0] + expr.oper.char + wrap[1]
        print_expr_tree(expr.left, " " * (len(s) + COL_SPACING))
        print(f'{s}')
        print_expr_tree(expr.right, " " * (len(s) + COL_SPACING))


def invert_expr_tree(expr):
    if isinstance(expr, Number):
        return
    
    elif isinstance(expr, UnaryExpr):
        invert_expr_tree(expr.expr)

    elif isinstance(expr, BinaryExpr):
        expr.left, expr.right = expr.right, expr.left
        invert_expr_tree(expr.left)
        invert_expr_tree(expr.right)


def invert_expr_tree_no_recursion(tree):

    stack = [(False, tree)]

    while stack:
        is_handled, node = stack.pop()
        if is_handled:
            if isinstance(node, BinaryExpr):
                node.left, node.right = node.right, node.left
        elif isinstance(node, Number):
            stack.append((True, node))
        elif isinstance(node, BinaryExpr):
            stack.append((True, node))
            stack.append((False, node.right))
            stack.append((False, node.left))
        else: # if isinstance(node, UnaryExpr):
            stack.append((False, node.expr))


def tree_to_str_expr(tree):

    stack = [(False, tree)]
    args = []

    while stack:
        is_handled, node = stack.pop()

        if is_handled:
            if isinstance(node, str):
                args.append(node)
            elif isinstance(node, Number):
                args.append(str(node))
            else:
                args.append(node.oper.char)
            continue
            
        if isinstance(node, Number) or isinstance(node, str):
            stack.append((True, node))
            continue
        elif isinstance(node, BinaryExpr):

            child_left_parenth = isinstance(node.left, _Operator) and node.left.oper.priority > node.oper.priority

            child_right_parenth = isinstance(node.right, _Operator) and node.right.oper.priority > node.oper.priority

            if child_right_parenth:
                stack.append((False, ')'))

            stack.append((False, node.right))

            if child_right_parenth:
                stack.append((False, '('))

            stack.append((True, node))

            if child_left_parenth:
                stack.append((False, ')'))

            stack.append((False, node.left))

            if child_left_parenth:
                stack.append((False, '('))

        elif isinstance(node, UnaryExpr):
            child_parenth = isinstance(node.expr, _Operator) and node.expr.oper.priority > node.oper.priority
            if child_parenth:
                stack.append((False, ')'))
            stack.append((False, node.expr))
            if child_parenth:
                stack.append((False, '('))
            stack.append((True, node))

        elif isinstance(node, WrapExpr):
            stack.append((False, node.oper.CLOSE))
            stack.append((False, node.expr))
            stack.append((False, node.oper.OPEN))

    return ' '.join(args)

    

def iter_tree(tree):

    stack = [(False, tree)]

    while stack:
        is_handled, node = stack.pop()

        if is_handled:
            yield node
        else:
            stack.append((True, node))

            if isinstance(node, Number):
                continue
            elif isinstance(node, BinaryExpr):
                stack.append((False, node.right))
                stack.append((False, node.left))
            else: # if isinstance(node, UnaryExpr):
                stack.append((False, node.expr))
    


@dataclass
class BinaryExpr:
    left: 'Expression'
    oper: BiOper
    right: 'Expression'

@dataclass
class UnaryExpr:
    oper: UnOper
    expr: 'Expression'

@dataclass
class WrapExpr:
    oper: _ExpressionOper
    expr: 'Expression'


Expression = TypeVar('Expression', bound=Union[BinaryExpr, UnaryExpr, Number])


def parse_expression(expr) -> Union[int, float]:
    if isinstance(expr, Number):
        return expr
    elif isinstance(expr, BinaryExpr):
        return expr.oper.func(parse_expression(expr.left), parse_expression(expr.right))
    else:
        return expr.oper.func(parse_expression(expr.expr))


class StringLexer(Enum):
    NONE = auto()
    NUMBER = auto()
    OPERAND = auto()

# TODO: Column Number
class LexerSyntaxError(Exception):
    pass

str_binary_opers = {oper.char: oper for oper in BiOper.__members__.values()}
str_unary_opers = {oper.char: oper for oper in UnOper.__members__.values()}

wrap_oper_openers = {Parenth.OPEN.char: Parenth.OPEN, Absolut.OPEN.char: Absolut.OPEN}


def lex_string_to_args(string) -> Tuple[List[Expression], List[int]]:

    def end_number():
        """Handles conversion of string number input to int or float and changes state"""
        nonlocal state
        if state == StringLexer.NUMBER:
            state = StringLexer.NONE
            num = args[-1]
            if '.' in num:
                args[-1] = float(num)
            else:
                args[-1] = int(num)

    def append(arg, loc=None):
        """So I don't forget to append to locs"""
        args.append(arg)
        locs.append(i if loc is None else loc)
    
    args: list = []
    locs: list = []
    state: StringLexer = StringLexer.NONE
    in_abs: bool = False

    char: str
    i: int
    for i, char in enumerate(string):
        if char.isspace():
            pass
        elif char.isdigit():
            if state is StringLexer.NUMBER:
                args[-1] += char
            else:
                append(char)
                state = StringLexer.NUMBER
        elif char == '.':
            if state is not StringLexer.NUMBER:
                append(char)
                state = StringLexer.NUMBER
            elif char not in args[-1]:
                args[-1] += char
            else:
                raise LexerSyntaxError("Multiple decimal points in value.")
        elif char == Absolut.CLOSE.char and in_abs:
            end_number()
            append(Absolut.CLOSE)
            state = StringLexer.OPERAND
        elif char == Parenth.CLOSE.char:
            end_number()
            append(Parenth.CLOSE)
            state = StringLexer.OPERAND
        elif wrap_op := wrap_oper_openers.get(char):
            if state is StringLexer.NUMBER or state is StringLexer.OPERAND:
                end_number()
                append(BiOper.TIMES)
            if wrap_op is Absolut.OPEN:
                in_abs = True
            state = StringLexer.NONE
            append(wrap_op)
        elif state is StringLexer.NONE and (unary_oper := str_unary_opers.get(char)):
            end_number()
            append(unary_oper)
        elif binary_oper := str_binary_opers.get(char):
            end_number()
            append(binary_oper)
            state = StringLexer.NONE
        else:
            raise LexerSyntaxError(f"Could not parse character: '{char}'")

    end_number()
    return args, locs

class StaticCheckError(Exception):
    
    def __init__(self, msg: str, loc: int = None):
        super().__init__(msg if loc is None else " "*loc+'^\n'+msg)


class Types(Enum):
    END = auto()
    OPERAND = auto()
    BINARY_OPERATOR = auto()
    UNARY_OPERATOR = auto()

    def __str__(self) -> str:
        return self._name_

class ExpectedError(StaticCheckError):

    def __init__(self, expected: List[Types], got: str, loc: int = None, msg: str = '') -> None:
        super().__init__(f"Expected {' or '.join(map(str, expected))}. Got `{got}`. {msg}", loc)

def static_check_args(args, locs):
    
    unclosed_parenthesis: int = 0
    unclosed_absolute: int = 0

    allowed_types: Set[Types] = {Types.OPERAND, Types.UNARY_OPERATOR}

    for expr, loc in zip(args, locs):
        if isinstance(expr, Number):
            if not Types.OPERAND in allowed_types:
                raise ExpectedError(allowed_types, expr, loc, "Did you miss an operator?")
            allowed_types = {Types.END, Types.BINARY_OPERATOR}

        elif isinstance(expr, BiOper):
            if not Types.BINARY_OPERATOR in allowed_types:
                raise ExpectedError(allowed_types, expr.char, loc, "Did you miss an operand?")
            allowed_types = {Types.OPERAND, Types.UNARY_OPERATOR}

        elif isinstance(expr, UnOper):
            if not Types.UNARY_OPERATOR in allowed_types:
                raise ExpectedError(allowed_types, expr.char, loc)
            allowed_types = {Types.UNARY_OPERATOR, Types.OPERAND}

        elif expr is Parenth.OPEN:
            if not Types.OPERAND in allowed_types:
                raise ExpectedError(allowed_types, expr.char, loc)
            unclosed_parenthesis += 1
            allowed_types = {Types.OPERAND, Types.UNARY_OPERATOR}

        elif expr is Parenth.CLOSE:
            if not Types.END in allowed_types:
                raise ExpectedError(allowed_types, expr.char, loc)
            if not unclosed_parenthesis:
                raise StaticCheckError("Missing opening `(` for closing `)`. Did you miss an open parenthesis?")
            unclosed_parenthesis -= 1
            allowed_types = {Types.BINARY_OPERATOR, Types.END}

        elif expr is Absolut.OPEN:
            if not Types.OPERAND in allowed_types:
                raise ExpectedError(allowed_types, expr.char, loc)
            unclosed_absolute += 1
            allowed_types = {Types.OPERAND, Types.UNARY_OPERATOR}

        elif expr is Absolut.CLOSE:
            if not Types.END in allowed_types:
                raise ExpectedError(allowed_types, expr.char, loc)
            if not unclosed_absolute:
                raise StaticCheckError("Missing opening `|` for closing `|`")
            unclosed_absolute -= 1
            allowed_types = {Types.BINARY_OPERATOR, Types.END}
        else:
            assert False, f"`{expr}` Not Implemented."

    last_char = str(args[-1]) if isinstance(args[-1], Number) else args[-1].char
    end_loc = locs[-1] + len(last_char)
    if Types.END not in allowed_types:
        raise ExpectedError(allowed_types, 'EOF', end_loc)
    if unclosed_absolute > 0:
        raise StaticCheckError(f"Expected {unclosed_absolute} closing `|`. Got EOF. Did you forget to close your absolute value expression?", end_loc)
    if unclosed_parenthesis > 0:
        raise StaticCheckError(f"Expected {unclosed_parenthesis} closing `)`. Got EOF. Did you forget to close a parenthesis?", end_loc)


def lex_args_to_tree(args: List[str]):
    if len(args) == 1:
        expr = args[0]
        if isinstance(expr, Number):
            return expr
        else:
            raise LexerSyntaxError(str(args))
    
    elif args[0] is Parenth.OPEN and args[-1] is Parenth.CLOSE:
        return lex_args_to_tree(args[1:-1])
    
    elif args[0] is Absolut.OPEN and args[-1] is Absolut.CLOSE:
        return WrapExpr(Absolut, lex_args_to_tree(args[1:-1]))

    best_priority, best_index = 0, -1

    if isinstance(args[0], UnOper):
        best_oper = args[0]
        best_index = 0

    depth = 0
    for i, arg in enumerate(args):
        if depth:
            if arg is Parenth.CLOSE or arg is Absolut.CLOSE:
                depth -= 1
        elif arg is Parenth.OPEN or arg is Absolut.OPEN:
            depth += 1
        elif isinstance(arg, BiOper):
            if arg.priority >= best_priority:
                best_priority, best_index = arg.priority, i
    
    if best_index == 0:
        # must be unary
        return UnaryExpr(oper=args[0], expr=lex_args_to_tree(args[1:]))

    left = lex_args_to_tree(args[:best_index])
    right = lex_args_to_tree(args[best_index+1:])
    oper = args[best_index]

    return BinaryExpr(
        left = left, 
        right = right,
        oper = oper,
    )


def main():
    print("\n====== CALCULATE ======")
    string = input("> ")
    try:
        args, locs = lex_string_to_args(string)
    except LexerSyntaxError as e:
        print(f"ERROR: {e}")
        return

    print("LEXED ARGUMENTS:", args)
    try:
        static_check_args(args, locs)
    except StaticCheckError as e:
        print("ERROR:")
        print(string)
        print(e, file=stderr)
        return

    expression = lex_args_to_tree(args)
    print(expression)
    print_expr_tree(expression)
    print("\nRESULT:", parse_expression(expression))

    print('\n===== Inverting Tree =====\n')
    invert_expr_tree_no_recursion(expression)
    print_expr_tree(expression)
    print("\nRESULT:", parse_expression(expression))

    print("CONVERTED BACK:", tree_to_str_expr(expression))
    return



if __name__ == "__main__":
    main()
    pass
