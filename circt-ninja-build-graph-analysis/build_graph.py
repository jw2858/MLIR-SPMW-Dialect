import re
from collections import Counter
from typing import Iterator, Optional, Sequence, Tuple

import networkx as nx
from split_command_line import split_command_line_posix

line_pattern = re.compile(r'^\[\d+/\d+\]')


def yield_command_args(ninja_output_file):
    with open(ninja_output_file, 'r') as f:
        for i, line in enumerate(f):
            if line_pattern.match(line) is not None:
                args = list(split_command_line_posix(line))
                num_args = len(args)
                if num_args > 5 and args[1] == ':' and args[2] == '&&' and args[-2] == '&&' and args[-1] == ':':
                    j = 3
                    current_args = []
                    while j < num_args - 2:
                        arg = args[j]
                        if arg == '&&':
                            if current_args:
                                yield current_args
                                current_args = []
                        else:
                            current_args.append(arg)
                        j += 1
                    if current_args:
                        yield current_args
                else:
                    j = 1
                    current_args = []
                    while j < num_args:
                        arg = args[j]
                        if arg == '&&':
                            if current_args:
                                yield current_args
                                current_args = []
                        else:
                            current_args.append(arg)
                        j += 1
                    if current_args:
                        yield current_args


class CXXCommandToken(object): pass


class CXXCommandPath(CXXCommandToken):
    """A path in a $CXX command"""
    __slots__ = ('path',)
    
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return self.__class__, (self.path,)
    
    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()
    
    def __hash__(self):
        return hash(self.__reduce__())
    
    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
            )


class CXXCommandFlag(CXXCommandToken):
    """A flag in a $CXX command"""
    pass


class Compile(CXXCommandFlag):
    """The -c flag"""
    def __reduce__(self):
        return self.__class__, ()
    
    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()
    
    def __hash__(self):
        return hash(self.__reduce__())
    
    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
        )


class Debug(CXXCommandFlag):
    """The -g flag"""
    def __reduce__(self):
        return self.__class__, ()
    
    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()
    
    def __hash__(self):
        return hash(self.__reduce__())
    
    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
        )


class Define(CXXCommandFlag):
    """The -D flag"""
    __slots__ = ('name', 'value')

    def __init__(self, name, value=None):
        self.name = name
        self.value = value

    def __reduce__(self):
        return self.__class__, (self.name, self.value)
    
    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()
    
    def __hash__(self):
        return hash(self.__reduce__())
    
    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
        )


class Force(CXXCommandFlag):
    """The -f flag"""
    __slots__ = ('force',)

    def __init__(self, force):
        self.force = force
        
    def __reduce__(self):
        return self.__class__, (self.force,)
    
    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()
    
    def __hash__(self):
        return hash(self.__reduce__())
    
    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
        )


class Include(CXXCommandFlag):
    """The -I flag"""
    __slots__ = ('include',)
    
    def __init__(self, include):
        self.include = include
    
    def __reduce__(self):
        return self.__class__, (self.include,)
    
    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()
    
    def __hash__(self):
        return hash(self.__reduce__())
    
    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
        )


class IncludeSystem(CXXCommandFlag):
    """The -isystem flag"""
    __slots__ = ('include_system',)
    
    def __init__(self, include_system):
        self.include_system = include_system
    
    def __reduce__(self):
        return self.__class__, (self.include_system,)
    
    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()
    
    def __hash__(self):
        return hash(self.__reduce__())
    
    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
        )


class Link(CXXCommandFlag):
    """The -l flag"""
    __slots__ = ('link',)
    
    def __init__(self, link):
        self.link = link
    
    def __reduce__(self):
        return self.__class__, (self.link,)
    
    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()
    
    def __hash__(self):
        return hash(self.__reduce__())
    
    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
        )


class Machine(CXXCommandFlag):
    """The -m flag"""
    __slots__ = ('machine',)
    
    def __init__(self, machine):
        self.machine = machine

    def __reduce__(self):
        return self.__class__, (self.machine,)
    
    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()
    
    def __hash__(self):
        return hash(self.__reduce__())
    
    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
        )


class Make(CXXCommandFlag):
    """The -MD flag"""    
    def __reduce__(self):
        return self.__class__, ()
    
    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()
    
    def __hash__(self):
        return hash(self.__reduce__())
    
    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
        )


class MakeTarget(CXXCommandFlag):
    """The -MT flag"""
    __slots__ = ('target',)
    
    def __init__(self, target):
        self.target = target
    
    def __reduce__(self):
        return self.__class__, (self.target,)
    
    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()
    
    def __hash__(self):
        return hash(self.__reduce__())
    
    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
        )


class MakeFile(CXXCommandFlag):
    """The -MF flag"""
    __slots__ = ('file',)
    
    def __init__(self, file):
        self.file = file

    def __reduce__(self):
        return self.__class__, (self.file,)
    
    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()
    
    def __hash__(self):
        return hash(self.__reduce__())
    
    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
        )


class Output(CXXCommandFlag):
    """The -o flag"""
    __slots__ = ('output',)

    def __init__(self, output):
        self.output = output
    
    def __reduce__(self):
        return self.__class__, (self.output,)
    
    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()
    
    def __hash__(self):
        return hash(self.__reduce__())
    
    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
        )


class Pedantic(CXXCommandFlag):
    """The -pedantic flag"""
    def __reduce__(self):
        return self.__class__, ()

    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()
    
    def __hash__(self):
        return hash(self.__reduce__())
    
    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
        )


class Standard(CXXCommandFlag):
    """The -std flag"""
    __slots__ = ('standard',)

    def __init__(self, standard):
        self.standard = standard
    
    def __reduce__(self):
        return self.__class__, (self.standard,)

    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()
    
    def __hash__(self):
        return hash(self.__reduce__())
    
    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
        )


class Warn(CXXCommandFlag):
    """The -W flag"""
    __slots__ = ('warn',)

    def __init__(self, warn):
        self.warn = warn
        
    def __reduce__(self):
        return self.__class__, (self.warn,)
    
    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()

    def __hash__(self):
        return hash(self.__reduce__())
    
    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
        )


def lex_cxx_command(
    args,  # type: Sequence[str]
):
    # type: (...) -> Iterator[CXXCommandToken]
    i = 0
    num_args = len(args)

    while i < num_args:
        arg = args[i]
        if not arg.startswith('-'):
            yield CXXCommandPath(arg)
        elif arg == '-c':
            yield Compile()
        elif arg.startswith('-D'):
            # -DNAME=VALUE or -DNAME
            rest = arg[2:]
            equals_indices = [
                i
                for i, char in enumerate(rest)
                if char == '='
            ]

            if not equals_indices:
                yield Define(rest)
            elif len(equals_indices) == 1:
                equals_index = equals_indices[0]
                name = rest[:equals_index]
                value = rest[equals_index + 1:]
                if name and value:
                    yield Define(name, value)
                else:
                    raise ValueError
            else:
                raise ValueError
        elif arg.startswith('-f'):
            force = arg[2:]
            if force:
                yield Force(force)
            else:
                raise ValueError
        elif arg == '-g':
            yield Debug()
        elif arg.startswith('-I'):
            include = arg[2:]
            if include:
                yield Include(include)
            else:
                raise ValueError
        elif arg == '-isystem':
            if i < num_args - 1 and not args[i + 1].startswith('-'):
                include_system = args[i + 1]
                yield IncludeSystem(include_system)
            else:
                raise ValueError
        elif arg.startswith('-l'):
            link = arg[2:]
            if link:
                yield Link(link)
            else:
                raise ValueError
        elif arg.startswith('-m'):
            machine = arg[2:]
            if machine:
                yield Machine(machine)
            else:
                raise ValueError
        elif arg == '-MD':
            yield Make()
        elif arg == '-MT':
            if i < num_args - 1 and not args[i + 1].startswith('-'):
                target = args[i + 1]
                yield MakeTarget(target)
                i += 1
            else:
                raise ValueError
        elif arg == '-MF':
            if i < num_args - 1 and not args[i + 1].startswith('-'):
                file = args[i + 1]
                yield MakeFile(file)
                i += 1
            else:
                raise ValueError
        elif arg == '-o':
            if i < num_args - 1 and not args[i + 1].startswith('-'):
                output = args[i + 1]
                yield Output(output)
                i += 1
            else:
                raise ValueError
        elif arg == '-pedantic':
            yield Pedantic()
        elif arg.startswith('-std='):
            standard = arg[5:]
            if standard:
                yield Standard(standard)
            else:
                raise ValueError
        elif arg.startswith('-W'):
            warn = arg[2:]
            if warn:
                yield Warn(warn)
            else:
                raise ValueError
        else:
            raise ValueError

        i += 1


class UnixBuildCommand(object): pass


class CXXCommand(UnixBuildCommand):
    __slots__ = (
        'cxx',
        'debug',
        'defines',
        'forces',
        'includes',
        'include_systems',
        'links',
        'machines',
        'makes',
        'output',
        'pedantic',
        'warns',
        'inputs',
        'standard',
    )

    def __init__(
        self,
        cxx,  # type: str
        debug,  # type: bool
        defines,  # type: Sequence[Tuple[str, Optional[str]]]
        forces,  # type: Sequence[str]
        includes,  # type: Sequence[str]
        include_systems,  # type: Sequence[str]
        links,  # type: Sequence[str]
        machines,  # type: str
        makes,  # type: Sequence[Tuple[str, str]]
        output,  # type: str
        pedantic,  # type: bool
        warns,  # type: Sequence[str]
        inputs,  # type: Sequence[str]
        standard,  # type: Optional[str]
    ):
        self.cxx = cxx
        self.debug = debug
        self.defines = defines
        self.forces = forces
        self.includes = includes
        self.include_systems = include_systems
        self.links = links
        self.machines = machines
        self.makes = makes
        self.output = output
        self.pedantic = pedantic
        self.warns = warns
        self.inputs = inputs
        self.standard = standard


    def __reduce__(self):
        return (
            self.__class__,
            (
                self.cxx,
                self.debug,
                self.defines,
                self.forces,
                self.includes,
                self.include_systems,
                self.links,
                self.machines,
                self.makes,
                self.output,
                self.pedantic,
                self.warns,
                self.inputs,
                self.standard,
            )

        )

    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()

    def __hash__(self):
        return hash(self.__reduce__())
    
    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
        )


def parse_cxx_command(
        args,  # type: Sequence[str]
):
    # type: (...) -> CXXCommand
    tokens = list(lex_cxx_command(args))
    if not tokens:
        raise ValueError

    if not isinstance(tokens[0], CXXCommandPath):
        raise ValueError

    cxx = tokens[0].path
    compile_ = False
    debug = False
    defines = []
    forces = []
    includes = []
    include_systems = []
    links = []
    machines = []
    makes = []
    output = None
    pedantic = False
    warns = []
    inputs = []
    standard = None

    i = 1
    num_tokens = len(tokens)

    while i < num_tokens:
        token = tokens[i]
        if isinstance(token, CXXCommandPath):
            inputs.append(token.path)
        elif isinstance(token, Compile):
            compile_ = True
        elif isinstance(token, Debug):
            debug = True
        elif isinstance(token, Define):
            defines.append((token.name, token.value))
        elif isinstance(token, Force):
            forces.append(token.force)
        elif isinstance(token, Include):
            includes.append(token.include)
        elif isinstance(token, IncludeSystem):
            include_systems.append(token.include_system)
        elif isinstance(token, Link):
            links.append(token.link)
        elif isinstance(token, Machine):
            machines.append(token.machine)
        elif isinstance(token, Make):
            if (
                    i < num_tokens - 2
                    and isinstance(tokens[i + 1], MakeTarget)
                    and isinstance(tokens[i + 2], MakeFile)
            ):
                makes.append(
                    (
                        tokens[i + 1].target,
                        tokens[i + 2].file
                    )
                )
                i += 2
            else:
                raise ValueError
        elif isinstance(token, Output):
            if output is None:
                output = token.output
            else:
                raise ValueError
        elif isinstance(token, Pedantic):
            pedantic = True
        elif isinstance(token, Standard):
            if standard is None:
                standard = token.standard
            else:
                raise ValueError
        elif isinstance(token, Warn):
            warns.append(token.warn)
        else:
            raise ValueError

        i += 1

    if output is None:
        raise ValueError

    if not inputs:
        raise ValueError

    return CXXCommand(
        cxx=cxx,
        debug=debug,
        defines=defines,
        forces=tuple(forces),
        includes=tuple(includes),
        include_systems=tuple(include_systems),
        links=tuple(links),
        machines=tuple(machines),
        makes=tuple(makes),
        output=output,
        pedantic=pedantic,
        warns=tuple(warns),
        inputs=tuple(inputs),
        standard=standard,
    )

class ARCommand(UnixBuildCommand):
    __slots__ = (
        'ar',
        'deterministic',
        'output',
        'quickly',
        'inputs',
    )

    def __init__(
            self,
            ar,  # type: str
            deterministic,  # type: bool
            output,  # type: str
            quickly,  # type: bool
            inputs,  # type: Sequence[str]
    ):
        self.ar = ar
        self.deterministic = deterministic
        self.output = output
        self.quickly = quickly
        self.inputs = inputs

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.ar,
                self.deterministic,
                self.output,
                self.quickly,
                self.inputs,
            )

        )

    def __eq__(self, other):
        return self.__reduce__() == other.__reduce__()

    def __hash__(self):
        return hash(self.__reduce__())

    def __repr__(self):
        cls, members = self.__reduce__()
        return '%s(%s)' % (
            cls.__name__,
            ', '.join(map(repr, members))
        )


def parse_ar_command(
        args,  # type: Sequence[str]
):
    if len(args) < 4:
        raise ValueError

    ar = args[0]

    options = set(args[1])
    if 'c' not in options:
        raise ValueError
    options.remove('c')

    if 'D' in options:
        deterministic = True
        options.remove('D')
    else:
        deterministic = False

    if 'q' in options:
        quickly = True
        options.remove('q')
    else:
        quickly = False
    
    output = args[2]
    inputs = args[3:]

    return ARCommand(
        ar=ar,
        deterministic=deterministic,
        output=output,
        quickly=quickly,
        inputs=inputs,
    )


def construct_build_graph(
        ninja_output_file,  # type: str
        cc,  # type: str
        cxx,  # type: str
        ar,  # type: str
):
    # type: (...) -> nx.DiGraph
    build_graph = nx.DiGraph()

    for args in yield_command_args(ninja_output_file):
        if args:
            if args[0] in (cc, cxx):
                cxx_command = parse_cxx_command(args)
                for input_ in cxx_command.inputs:
                    build_graph.add_edge(input_, cxx_command.output)
            elif args[0] == ar:
                ar_command = parse_ar_command(args)
                for input_ in ar_command.inputs:
                    build_graph.add_edge(input_, ar_command.output)

    return build_graph
