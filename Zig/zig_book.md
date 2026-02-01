# easy zig

*A simple, practical guide to the Zig programming language*

---

## Table of Contents

- [0. Hello World](#0-hello-world)
- [1. Introduction to Zig](#1-introduction-to-zig)
- [2. Variables and Types](#2-variables-and-types)
- [3. Control Flow](#3-control-flow)
- [4. Functions](#4-functions)
- [5. Arrays and Slices](#5-arrays-and-slices)
- [6. Structs](#6-structs)
- [7. Enums and Unions](#7-enums-and-unions)
- [8. Error Handling](#8-error-handling)
- [9. Optionals](#9-optionals)
- [10. Pointers](#10-pointers)
- [11. Memory Allocation](#11-memory-allocation)
- [12. Compile-Time Execution](#12-compile-time-execution)
- [13. Testing](#13-testing)
- [14. Build System](#14-build-system)

---

## 0. Hello World


Let's start with the simplest Zig program and then explore memory allocation.

## How to Run

Save your code to a file (e.g., `hello.zig`) and run it with:

```bash
zig run hello.zig
```

That's it! Zig compiles and runs your program in one command.

## Other Ways to Build

```
┌─────────────────────────────────────────┐
│  zig run file.zig      # Compile & run  │
│  zig build-exe file.zig # Just compile  │
│  ./file                 # Run binary    │
└─────────────────────────────────────────┘
```


```
Running Zig Programs:

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  hello.zig  │ -> │  zig run    │ -> │   Output    │
│  (source)   │    │  (compile)  │    │  (result)   │
└─────────────┘    └─────────────┘    └─────────────┘

Memory Management with defer:

    ┌────────────────────────────────┐
    │  const x = try allocate();     │
    │  defer free(x);  <- registered │
    │                                │
    │  ... use x ...                 │
    │                                │
    └────────────────────────────────┘
              │
              ▼ (scope ends)
    ┌────────────────────────────────┐
    │  free(x) <- executed!          │
    └────────────────────────────────┘
```


### Example: hello_world

The simplest Zig program - prints Hello World to the console

```zig
const std = @import("std");

pub fn main() void {
    std.debug.print("Hello, World!\n", .{});
}
```

**Output:**
```
Hello, World!
```


### Example: allocator_example

A comprehensive example showing memory allocation with defer for automatic cleanup

```zig
const std = @import("std");

// Function that takes an allocator and creates a dynamic string
fn createGreeting(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    // Allocate memory for the greeting
    const greeting = try std.fmt.allocPrint(allocator, "Hello, {s}! Welcome to Zig.", .{name});
    return greeting;
}

// Function that creates a dynamic array
fn createNumbers(allocator: std.mem.Allocator, count: usize) ![]u32 {
    // Allocate an array of u32
    const numbers = try allocator.alloc(u32, count);

    // Fill with values
    for (numbers, 0..) |*num, i| {
        num.* = @intCast(i * 10);
    }

    return numbers;
}

pub fn main() !void {
    // Create a General Purpose Allocator (GPA)
    // GPA detects memory leaks and double-frees in debug mode
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const status = gpa.deinit();
        if (status == .leak) {
            std.debug.print("Memory leak detected!\n", .{});
        }
    }

    const allocator = gpa.allocator();

    // Example 1: Using defer for automatic cleanup
    std.debug.print("=== Example 1: String with defer ===\n", .{});
    {
        const greeting = try createGreeting(allocator, "Alice");
        defer allocator.free(greeting); // Automatically freed when scope ends

        std.debug.print("{s}\n", .{greeting});
    } // greeting is freed here

    // Example 2: Dynamic array with defer
    std.debug.print("\n=== Example 2: Array with defer ===\n", .{});
    {
        const numbers = try createNumbers(allocator, 5);
        defer allocator.free(numbers); // Automatically freed when scope ends

        std.debug.print("Numbers: ", .{});
        for (numbers) |n| {
            std.debug.print("{d} ", .{n});
        }
        std.debug.print("\n", .{});
    } // numbers is freed here

    // Example 3: Multiple allocations with defer (LIFO order)
    std.debug.print("\n=== Example 3: Multiple defers (LIFO order) ===\n", .{});
    {
        const str1 = try createGreeting(allocator, "Bob");
        defer {
            std.debug.print("Freeing str1\n", .{});
            allocator.free(str1);
        }

        const str2 = try createGreeting(allocator, "Charlie");
        defer {
            std.debug.print("Freeing str2\n", .{});
            allocator.free(str2);
        }

        std.debug.print("{s}\n", .{str1});
        std.debug.print("{s}\n", .{str2});
    } // str2 freed first, then str1 (LIFO)

    // Example 4: Using ArrayListUnmanaged with allocator
    std.debug.print("\n=== Example 4: ArrayListUnmanaged ===\n", .{});
    {
        var list: std.ArrayListUnmanaged(i32) = .{};
        defer list.deinit(allocator); // Pass allocator to deinit

        try list.append(allocator, 100);
        try list.append(allocator, 200);
        try list.append(allocator, 300);

        std.debug.print("ArrayList items: ", .{});
        for (list.items) |item| {
            std.debug.print("{d} ", .{item});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("\n=== All memory properly freed! ===\n", .{});
}
```

**Output:**
```
=== Example 1: String with defer ===
Hello, Alice! Welcome to Zig.

=== Example 2: Array with defer ===
Numbers: 0 10 20 30 40 

=== Example 3: Multiple defers (LIFO order) ===
Hello, Bob! Welcome to Zig.
Hello, Charlie! Welcome to Zig.
Freeing str2
Freeing str1

=== Example 4: ArrayListUnmanaged ===
ArrayList items: 100 200 300 

=== All memory properly freed! ===
```


---

## 1. Introduction to Zig


Zig is a modern systems programming language designed for:
- **Performance**: No hidden control flow or allocations
- **Safety**: Optional safety checks, no undefined behavior
- **Simplicity**: No hidden magic, readable code
- **Interoperability**: Direct C ABI compatibility

## Getting Started

Install Zig from https://ziglang.org/download/

```
┌─────────────────────────────────────────┐
│           Zig Workflow                  │
├─────────────────────────────────────────┤
│  1. Write code  →  main.zig             │
│  2. Compile     →  zig build-exe        │
│  3. Run         →  ./main               │
│                                         │
│  Or simply:     →  zig run main.zig     │
└─────────────────────────────────────────┘
```


```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Source     │ -> │   Compiler   │ -> │   Binary     │
│  main.zig    │    │     zig      │    │   ./main     │
└──────────────┘    └──────────────┘    └──────────────┘
```


### Example: hello_world

The classic Hello World program

```zig
const std = @import("std");

// This is your first Zig program!
// std.debug.print outputs to stderr
pub fn main() void {
    std.debug.print("Hello, Zig!\n", .{});
}
```

**Output:**
```
Hello, Zig!
```


---

## 2. Variables and Types


Zig has a simple type system with explicit types.

## Variable Declaration

```
┌─────────────────────────────────────────┐
│  const x = 5;    // Immutable (const)   │
│  var y = 10;     // Mutable (var)       │
│  var z: i32 = 0; // Explicit type       │
└─────────────────────────────────────────┘
```

## Basic Types

| Type    | Description          | Size    |
|---------|----------------------|---------|
| i8-i128 | Signed integers      | 1-16 B  |
| u8-u128 | Unsigned integers    | 1-16 B  |
| f32,f64 | Floating point       | 4,8 B   |
| bool    | Boolean              | 1 B     |
| void    | No value             | 0 B     |


```
Memory Layout (Little Endian):

u8:   [xxxxxxxx]           (1 byte)
u16:  [xxxxxxxx][xxxxxxxx] (2 bytes)
u32:  [xxxx][xxxx][xxxx][xxxx] (4 bytes)

i8 range:  -128 to 127
u8 range:  0 to 255
```


### Example: variables

Variable declaration and basic types

```zig
const std = @import("std");

pub fn main() void {
    // Immutable constant
    const message = "Hello";

    // Mutable variable
    var count: i32 = 0;
    count += 1;

    // Type inference
    const pi = 3.14159;

    // Explicit type
    const byte: u8 = 255;

    std.debug.print("message: {s}\n", .{message});
    std.debug.print("count: {}\n", .{count});
    std.debug.print("pi: {d:.2}\n", .{pi});
    std.debug.print("byte: {}\n", .{byte});
}
```

**Output:**
```
message: Hello
count: 1
pi: 3.14
byte: 255
```


### Example: type_coercion

Type conversions and coercion

```zig
const std = @import("std");

pub fn main() void {
    // Integer types
    const small: u8 = 10;
    const big: u32 = small; // Safe widening

    // Explicit cast for narrowing
    const large: u32 = 1000;
    const truncated: u8 = @truncate(large);

    std.debug.print("small: {}, big: {}\n", .{ small, big });
    std.debug.print("large: {}, truncated: {}\n", .{ large, truncated });
}
```

**Output:**
```
small: 10, big: 10
large: 1000, truncated: 232
```


---

## 3. Control Flow


Zig provides standard control flow with some unique features.

## If/Else

```
┌─────────────────────────────────────────┐
│  if (condition) {                       │
│      // true branch                     │
│  } else {                               │
│      // false branch                    │
│  }                                      │
└─────────────────────────────────────────┘
```

## While Loop

```
┌─────────────────────────────────────────┐
│  while (condition) : (continue_expr) {  │
│      // loop body                       │
│  }                                      │
└─────────────────────────────────────────┘
```


```
Control Flow Diagram:

    ┌───────┐
    │ Start │
    └───┬───┘
        │
    ┌───▼───┐     Yes    ┌───────┐
    │ cond? ├────────────► Block │
    └───┬───┘            └───────┘
        │ No
    ┌───▼───┐
    │  End  │
    └───────┘
```


### Example: if_else

If/else statements and expressions

```zig
const std = @import("std");

pub fn main() void {
    const x: i32 = 42;

    // Basic if/else
    if (x > 0) {
        std.debug.print("x is positive\n", .{});
    } else if (x < 0) {
        std.debug.print("x is negative\n", .{});
    } else {
        std.debug.print("x is zero\n", .{});
    }

    // If as expression
    const abs_x = if (x < 0) -x else x;
    std.debug.print("abs(x) = {}\n", .{abs_x});
}
```

**Output:**
```
x is positive
abs(x) = 42
```


### Example: loops

While and for loops

```zig
const std = @import("std");

pub fn main() void {
    // While loop
    var i: u32 = 0;
    while (i < 3) : (i += 1) {
        std.debug.print("while i = {}\n", .{i});
    }

    // For loop over range
    for (0..3) |j| {
        std.debug.print("for j = {}\n", .{j});
    }

    // For loop over array
    const arr = [_]i32{ 10, 20, 30 };
    for (arr) |val| {
        std.debug.print("val = {}\n", .{val});
    }
}
```

**Output:**
```
while i = 0
while i = 1
while i = 2
for j = 0
for j = 1
for j = 2
val = 10
val = 20
val = 30
```


### Example: switch

Switch expressions

```zig
const std = @import("std");

pub fn main() void {
    const day: u8 = 3;

    // Switch statement
    const name = switch (day) {
        1 => "Monday",
        2 => "Tuesday",
        3 => "Wednesday",
        4 => "Thursday",
        5 => "Friday",
        6, 7 => "Weekend",
        else => "Invalid",
    };

    std.debug.print("Day {}: {s}\n", .{ day, name });
}
```

**Output:**
```
Day 3: Wednesday
```


---

## 4. Functions


Functions are first-class values in Zig.

## Function Syntax

```
┌─────────────────────────────────────────┐
│  fn name(param: Type) ReturnType {      │
│      return value;                      │
│  }                                      │
└─────────────────────────────────────────┘
```

## Special Features
- Functions can be passed as values
- Comptime function evaluation
- Inline functions with `inline`


```
Function Call Stack:

┌─────────────────────┐
│   main()            │  <- Stack Frame
├─────────────────────┤
│   add(3, 4)         │  <- New Frame
│   a = 3, b = 4      │
│   return 7          │
├─────────────────────┤
│   (returned to main)│
└─────────────────────┘
```


### Example: basic_functions

Basic function definitions

```zig
const std = @import("std");

// Simple function
fn add(a: i32, b: i32) i32 {
    return a + b;
}

// Function with multiple return (tuple)
fn divmod(a: i32, b: i32) struct { quot: i32, rem: i32 } {
    return .{ .quot = @divTrunc(a, b), .rem = @mod(a, b) };
}

pub fn main() void {
    const sum = add(3, 4);
    std.debug.print("3 + 4 = {}\n", .{sum});

    const result = divmod(17, 5);
    std.debug.print("17 / 5 = {} remainder {}\n", .{ result.quot, result.rem });
}
```

**Output:**
```
3 + 4 = 7
17 / 5 = 3 remainder 2
```


### Example: function_pointers

Function pointers and higher-order functions

```zig
const std = @import("std");

fn double(x: i32) i32 {
    return x * 2;
}

fn triple(x: i32) i32 {
    return x * 3;
}

// Function that takes a function pointer
fn apply(value: i32, func: *const fn (i32) i32) i32 {
    return func(value);
}

pub fn main() void {
    const x: i32 = 5;

    std.debug.print("double({}) = {}\n", .{ x, apply(x, &double) });
    std.debug.print("triple({}) = {}\n", .{ x, apply(x, &triple) });
}
```

**Output:**
```
double(5) = 10
triple(5) = 15
```


---

## 5. Arrays and Slices


Arrays have fixed size; slices are views into arrays.

## Arrays vs Slices

```
┌─────────────────────────────────────────┐
│  Array: [N]T  - Fixed size at compile   │
│  Slice: []T   - Runtime view into array │
└─────────────────────────────────────────┘
```


```
Array in Memory:
┌───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │  arr[5]i32
└───┴───┴───┴───┴───┘
  0   1   2   3   4

Slice (view):
        ┌───────────┐
        │ ptr + len │
        └─────┬─────┘
              │
┌───┬───┬───┬─▼─┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │
└───┴───┴───┴───┴───┘
        └─────┘
        slice [1..4]
```


### Example: arrays

Array basics

```zig
const std = @import("std");

pub fn main() void {
    // Array with explicit size
    const arr1: [5]i32 = [_]i32{ 1, 2, 3, 4, 5 };

    // Array with inferred size
    const arr2 = [_]i32{ 10, 20, 30 };

    // Access elements
    std.debug.print("arr1[0] = {}\n", .{arr1[0]});
    std.debug.print("arr2.len = {}\n", .{arr2.len});

    // Iterate
    var sum: i32 = 0;
    for (arr1) |x| {
        sum += x;
    }
    std.debug.print("sum of arr1 = {}\n", .{sum});
}
```

**Output:**
```
arr1[0] = 1
arr2.len = 3
sum of arr1 = 15
```


### Example: slices

Slices - views into arrays

```zig
const std = @import("std");

pub fn main() void {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };

    // Create slice from array
    const slice: []const i32 = arr[1..4]; // [2, 3, 4]

    std.debug.print("slice.len = {}\n", .{slice.len});

    for (slice, 0..) |val, i| {
        std.debug.print("slice[{}] = {}\n", .{ i, val });
    }
}
```

**Output:**
```
slice.len = 3
slice[0] = 2
slice[1] = 3
slice[2] = 4
```


### Example: string_literals

Strings as byte slices

```zig
const std = @import("std");

pub fn main() void {
    // String literal is []const u8
    const hello = "Hello, World!";

    std.debug.print("String: {s}\n", .{hello});
    std.debug.print("Length: {}\n", .{hello.len});
    std.debug.print("First char: {c}\n", .{hello[0]});

    // Iterate over characters
    for (hello[0..5]) |c| {
        std.debug.print("{c} ", .{c});
    }
    std.debug.print("\n", .{});
}
```

**Output:**
```
String: Hello, World!
Length: 13
First char: H
H e l l o
```


---

## 6. Structs


Structs group related data together.

## Struct Definition

```
┌─────────────────────────────────────────┐
│  const Point = struct {                 │
│      x: i32,                            │
│      y: i32,                            │
│                                         │
│      fn init(x: i32, y: i32) Point {    │
│          return .{ .x = x, .y = y };    │
│      }                                  │
│  };                                     │
└─────────────────────────────────────────┘
```


```
Struct Memory Layout:

Point { x: i32, y: i32 }

┌─────────────┬─────────────┐
│    x: i32   │    y: i32   │
│   4 bytes   │   4 bytes   │
└─────────────┴─────────────┘
     Total: 8 bytes
```


### Example: basic_struct

Struct definition with methods

```zig
const std = @import("std");

// Define a struct
const Point = struct {
    x: i32,
    y: i32,

    // Method
    fn distance_from_origin(self: Point) f32 {
        const fx: f32 = @floatFromInt(self.x);
        const fy: f32 = @floatFromInt(self.y);
        return @sqrt(fx * fx + fy * fy);
    }
};

pub fn main() void {
    // Create instance
    const p = Point{ .x = 3, .y = 4 };

    std.debug.print("Point: ({}, {})\n", .{ p.x, p.y });
    std.debug.print("Distance: {d:.2}\n", .{p.distance_from_origin()});
}
```

**Output:**
```
Point: (3, 4)
Distance: 5.00
```


### Example: struct_defaults

Struct with default field values

```zig
const std = @import("std");

const Config = struct {
    // Fields with default values
    width: u32 = 800,
    height: u32 = 600,
    title: []const u8 = "Untitled",
};

pub fn main() void {
    // Use all defaults
    const c1 = Config{};

    // Override some defaults
    const c2 = Config{ .width = 1920, .height = 1080 };

    std.debug.print("c1: {}x{} '{s}'\n", .{ c1.width, c1.height, c1.title });
    std.debug.print("c2: {}x{} '{s}'\n", .{ c2.width, c2.height, c2.title });
}
```

**Output:**
```
c1: 800x600 'Untitled'
c2: 1920x1080 'Untitled'
```


---

## 7. Enums and Unions


Enums define a set of named values. Tagged unions combine enums with data.

## Enum

```
┌─────────────────────────────────────────┐
│  const Color = enum { red, green, blue };│
└─────────────────────────────────────────┘
```

## Tagged Union

```
┌─────────────────────────────────────────┐
│  const Value = union(enum) {            │
│      int: i32,                          │
│      float: f64,                        │
│      none,                              │
│  };                                     │
└─────────────────────────────────────────┘
```


```
Tagged Union Memory:

Value union(enum):
┌─────────┬─────────────────┐
│   Tag   │      Data       │
│ (enum)  │ (largest field) │
└─────────┴─────────────────┘

Tag values: .int=0, .float=1, .boolean=2, .none=3
```


### Example: enums

Enums with methods

```zig
const std = @import("std");

const Direction = enum {
    north,
    south,
    east,
    west,

    // Enum method
    fn opposite(self: Direction) Direction {
        return switch (self) {
            .north => .south,
            .south => .north,
            .east => .west,
            .west => .east,
        };
    }
};

pub fn main() void {
    const dir = Direction.north;
    const opp = dir.opposite();

    std.debug.print("Direction: {}\n", .{dir});
    std.debug.print("Opposite: {}\n", .{opp});
}
```

**Output:**
```
Direction: .north
Opposite: .south
```


### Example: tagged_union

Tagged unions for type-safe variants

```zig
const std = @import("std");

const Value = union(enum) {
    int: i32,
    float: f64,
    boolean: bool,
    none: void,
};

fn printValue(val: Value) void {
    switch (val) {
        .int => |i| std.debug.print("int: {}\n", .{i}),
        .float => |f| std.debug.print("float: {d:.2}\n", .{f}),
        .boolean => |b| std.debug.print("bool: {}\n", .{b}),
        .none => std.debug.print("none\n", .{}),
    }
}

pub fn main() void {
    const v1 = Value{ .int = 42 };
    const v2 = Value{ .float = 3.14 };
    const v3 = Value{ .boolean = true };
    const v4 = Value{ .none = {} };

    printValue(v1);
    printValue(v2);
    printValue(v3);
    printValue(v4);
}
```

**Output:**
```
int: 42
float: 3.14
bool: true
none
```


---

## 8. Error Handling


Zig uses explicit error handling with error unions.

## Error Union Type

```
┌─────────────────────────────────────────┐
│  fn divide(a: i32, b: i32) !i32 {       │
│      if (b == 0) return error.DivByZero;│
│      return @divTrunc(a, b);            │
│  }                                      │
└─────────────────────────────────────────┘
```

## Error Handling Options
- `try`: Propagate error up
- `catch`: Handle error
- `orelse`: Default value on error


```
Error Union Type:

    anyerror!T or ErrorSet!T
         │
    ┌────┴────┐
    │         │
  error     value
    │         │
┌───▼───┐ ┌───▼───┐
│ catch │ │ success│
└───────┘ └───────┘
```


### Example: error_basics

Basic error handling with catch

```zig
const std = @import("std");

const MathError = error{
    DivisionByZero,
    Overflow,
};

fn divide(a: i32, b: i32) MathError!i32 {
    if (b == 0) return MathError.DivisionByZero;
    return @divTrunc(a, b);
}

pub fn main() void {
    // Using catch to handle error
    const result1 = divide(10, 2) catch |err| {
        std.debug.print("Error: {}\n", .{err});
        return;
    };
    std.debug.print("10 / 2 = {}\n", .{result1});

    // Using catch with default
    const result2 = divide(10, 0) catch 0;
    std.debug.print("10 / 0 = {} (default)\n", .{result2});
}
```

**Output:**
```
10 / 2 = 5
10 / 0 = 0 (default)
```


### Example: try_keyword

Using try to propagate errors

```zig
const std = @import("std");

const ParseError = error{InvalidChar};

fn parseDigit(c: u8) ParseError!u8 {
    if (c >= '0' and c <= '9') {
        return c - '0';
    }
    return ParseError.InvalidChar;
}

fn parseNumber(s: []const u8) ParseError!u32 {
    var result: u32 = 0;
    for (s) |c| {
        // try propagates error if parseDigit fails
        const digit = try parseDigit(c);
        result = result * 10 + digit;
    }
    return result;
}

pub fn main() void {
    if (parseNumber("123")) |num| {
        std.debug.print("Parsed: {}\n", .{num});
    } else |err| {
        std.debug.print("Parse error: {}\n", .{err});
    }

    if (parseNumber("12x")) |num| {
        std.debug.print("Parsed: {}\n", .{num});
    } else |err| {
        std.debug.print("Parse error: {}\n", .{err});
    }
}
```

**Output:**
```
Parsed: 123
Parse error: error.InvalidChar
```


---

## 9. Optionals


Optionals represent values that may or may not exist.

## Optional Type

```
┌─────────────────────────────────────────┐
│  var x: ?i32 = null;   // No value      │
│  x = 42;               // Has value     │
│                                         │
│  if (x) |val| {        // Unwrap        │
│      // use val                         │
│  }                                      │
└─────────────────────────────────────────┘
```


```
Optional Type ?T:

┌─────────────┐
│   ?i32      │
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
 null    value
   │       │
┌──▼──┐ ┌──▼──┐
│empty│ │ 42  │
└─────┘ └─────┘
```


### Example: optionals

Optional types and unwrapping

```zig
const std = @import("std");

fn find_first_even(arr: []const i32) ?i32 {
    for (arr) |x| {
        if (@mod(x, 2) == 0) return x;
    }
    return null; // Not found
}

pub fn main() void {
    const arr1 = [_]i32{ 1, 3, 4, 7, 9 };
    const arr2 = [_]i32{ 1, 3, 5, 7, 9 };

    // Using if to unwrap
    if (find_first_even(&arr1)) |val| {
        std.debug.print("First even: {}\n", .{val});
    } else {
        std.debug.print("No even number found\n", .{});
    }

    // Using orelse for default
    const result = find_first_even(&arr2) orelse -1;
    std.debug.print("Result with default: {}\n", .{result});
}
```

**Output:**
```
First even: 4
Result with default: -1
```


### Example: optional_pointers

Optional pointers for linked structures

```zig
const std = @import("std");

const Node = struct {
    value: i32,
    next: ?*Node, // Optional pointer
};

pub fn main() void {
    var node3 = Node{ .value = 30, .next = null };
    var node2 = Node{ .value = 20, .next = &node3 };
    var node1 = Node{ .value = 10, .next = &node2 };

    // Traverse linked list
    var current: ?*Node = &node1;
    while (current) |node| {
        std.debug.print("Value: {}\n", .{node.value});
        current = node.next;
    }
}
```

**Output:**
```
Value: 10
Value: 20
Value: 30
```


---

## 10. Pointers


Zig has different pointer types for different use cases.

## Pointer Types

```
┌─────────────────────────────────────────┐
│  *T       - Single-item pointer         │
│  [*]T     - Many-item pointer           │
│  *const T - Pointer to const            │
│  ?*T      - Optional pointer            │
└─────────────────────────────────────────┘
```


```
Pointer to Variable:

   ptr          x
┌───────┐   ┌───────┐
│ addr ─┼──►│  10   │
└───────┘   └───────┘

*ptr dereferences to get value
&x gets address of x
```


### Example: pointers_basic

Basic pointer operations

```zig
const std = @import("std");

pub fn main() void {
    var x: i32 = 10;

    // Get pointer
    const ptr: *i32 = &x;

    // Dereference
    std.debug.print("x = {}\n", .{x});
    std.debug.print("*ptr = {}\n", .{ptr.*});

    // Modify through pointer
    ptr.* = 20;
    std.debug.print("After modification: x = {}\n", .{x});
}
```

**Output:**
```
x = 10
*ptr = 10
After modification: x = 20
```


### Example: pointer_arithmetic

Pointer arithmetic and slicing

```zig
const std = @import("std");

pub fn main() void {
    const arr = [_]i32{ 10, 20, 30, 40, 50 };

    // Many-item pointer
    const ptr: [*]const i32 = &arr;

    // Access by index
    std.debug.print("ptr[0] = {}\n", .{ptr[0]});
    std.debug.print("ptr[2] = {}\n", .{ptr[2]});

    // Slice from pointer
    const slice = ptr[1..4];
    for (slice) |val| {
        std.debug.print("slice val: {}\n", .{val});
    }
}
```

**Output:**
```
ptr[0] = 10
ptr[2] = 30
slice val: 20
slice val: 30
slice val: 40
```


---

## 11. Memory Allocation


Zig gives you explicit control over memory allocation.

## Allocator Interface

```
┌─────────────────────────────────────────┐
│  const allocator = std.heap.page_allocator;│
│  const ptr = try allocator.create(T);   │
│  defer allocator.destroy(ptr);          │
└─────────────────────────────────────────┘
```

## Common Allocators
- `page_allocator`: OS page allocation
- `GeneralPurposeAllocator`: Debug-friendly
- `ArenaAllocator`: Bulk deallocation


```
Memory Management:

Stack (automatic):
┌─────────────────┐
│ local variables │ <- fast, auto cleanup
└─────────────────┘

Heap (manual):
┌─────────────────┐
│ allocator.alloc │ <- explicit alloc
│ allocator.free  │ <- explicit free
└─────────────────┘

Use defer for cleanup!
```


### Example: allocation

Dynamic memory allocation

```zig
const std = @import("std");

pub fn main() !void {
    // Use general purpose allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Allocate array
    const arr = try allocator.alloc(i32, 5);
    defer allocator.free(arr);

    // Initialize
    for (arr, 0..) |*item, i| {
        item.* = @intCast(i * 10);
    }

    // Print
    for (arr) |val| {
        std.debug.print("{} ", .{val});
    }
    std.debug.print("\n", .{});
}
```

**Output:**
```
0 10 20 30 40
```


### Example: arraylist

Using ArrayList for dynamic arrays

```zig
const std = @import("std");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // ArrayList - dynamic array
    var list: std.ArrayListUnmanaged(i32) = .empty;
    defer list.deinit(allocator);

    // Add elements
    try list.append(allocator, 10);
    try list.append(allocator, 20);
    try list.append(allocator, 30);

    std.debug.print("List: ", .{});
    for (list.items) |val| {
        std.debug.print("{} ", .{val});
    }
    std.debug.print("\n", .{});
    std.debug.print("Length: {}\n", .{list.items.len});
}
```

**Output:**
```
List: 10 20 30 
Length: 3
```


---

## 12. Compile-Time Execution


Zig can execute code at compile time using `comptime`.

## Comptime Features

```
┌─────────────────────────────────────────┐
│  comptime {                             │
│      // Runs at compile time            │
│  }                                      │
│                                         │
│  fn generic(comptime T: type) type {    │
│      // Type-level computation          │
│  }                                      │
└─────────────────────────────────────────┘
```


```
Comptime vs Runtime:

┌──────────────────┐     ┌──────────────────┐
│   Compile Time   │     │    Run Time      │
├──────────────────┤     ├──────────────────┤
│ comptime blocks  │     │ Regular code     │
│ Type computation │     │ User input       │
│ Generic params   │     │ Dynamic data     │
│ Const folding    │     │ Heap allocation  │
└──────────────────┘     └──────────────────┘
        │                        │
        └────────┬───────────────┘
                 │
            Final Binary
```


### Example: comptime_basic

Compile-time computation

```zig
const std = @import("std");

// Compile-time function
fn factorial(n: u64) u64 {
    if (n == 0) return 1;
    return n * factorial(n - 1);
}

// Const values with comptime-known args are computed at compile time
const factorial_10 = factorial(10);

pub fn main() void {
    // This value is computed at compile time!
    std.debug.print("10! = {}\n", .{factorial_10});

    // Comptime type introspection
    const T = i32;
    std.debug.print("Size of i32: {} bytes\n", .{@sizeOf(T)});

    // Inline comptime block
    const squares = comptime blk: {
        var arr: [5]i32 = undefined;
        for (0..5) |i| {
            arr[i] = @as(i32, @intCast(i)) * @as(i32, @intCast(i));
        }
        break :blk arr;
    };
    std.debug.print("Squares: ", .{});
    for (squares) |s| {
        std.debug.print("{} ", .{s});
    }
    std.debug.print("\n", .{});
}
```

**Output:**
```
10! = 3628800
Size of i32: 4 bytes
Squares: 0 1 4 9 16
```


### Example: generic_function

Generic functions and types with comptime

```zig
const std = @import("std");

// Generic max function using comptime
fn max(comptime T: type, a: T, b: T) T {
    return if (a > b) a else b;
}

// Generic struct
fn Pair(comptime T: type) type {
    return struct {
        first: T,
        second: T,

        fn swap(self: *@This()) void {
            const tmp = self.first;
            self.first = self.second;
            self.second = tmp;
        }
    };
}

pub fn main() void {
    // Works with any comparable type
    std.debug.print("max(3, 7) = {}\n", .{max(i32, 3, 7)});
    std.debug.print("max(3.5, 2.1) = {d:.1}\n", .{max(f64, 3.5, 2.1)});

    // Generic struct usage
    var pair = Pair(i32){ .first = 10, .second = 20 };
    std.debug.print("Before swap: {}, {}\n", .{ pair.first, pair.second });
    pair.swap();
    std.debug.print("After swap: {}, {}\n", .{ pair.first, pair.second });
}
```

**Output:**
```
max(3, 7) = 7
max(3.5, 2.1) = 3.5
Before swap: 10, 20
After swap: 20, 10
```


---

## 13. Testing


Zig has built-in testing support.

## Test Syntax

```
┌─────────────────────────────────────────┐
│  test "description" {                   │
│      try std.testing.expect(true);      │
│      try std.testing.expectEqual(1, 1); │
│  }                                      │
└─────────────────────────────────────────┘
```

Run tests with: `zig test file.zig`


```
Test Workflow:

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Write test │ -> │  zig test   │ -> │   Results   │
└─────────────┘    └─────────────┘    └─────────────┘
                          │
                   ┌──────┴──────┐
                   │             │
                 Pass          Fail
                   │             │
                  [OK]      [Error msg]
```


### Example: testing

Writing and running tests

```zig
const std = @import("std");

fn add(a: i32, b: i32) i32 {
    return a + b;
}

fn divide(a: i32, b: i32) !i32 {
    if (b == 0) return error.DivisionByZero;
    return @divTrunc(a, b);
}

test "add function" {
    try std.testing.expectEqual(add(2, 3), 5);
    try std.testing.expectEqual(add(-1, 1), 0);
}

test "divide function" {
    const result = try divide(10, 2);
    try std.testing.expectEqual(result, 5);
}

test "divide by zero returns error" {
    const result = divide(10, 0);
    try std.testing.expectError(error.DivisionByZero, result);
}

pub fn main() void {
    std.debug.print("Run 'zig test' to execute tests\n", .{});
    std.debug.print("add(2,3) = {}\n", .{add(2, 3)});
}
```

**Output:**
```
Run 'zig test' to execute tests
add(2,3) = 5
```


---

## 14. Build System


Zig has a powerful built-in build system using `build.zig`.

## Basic build.zig

```
┌─────────────────────────────────────────┐
│  zig init     - Create new project      │
│  zig build    - Build the project       │
│  zig build run - Build and run          │
└─────────────────────────────────────────┘
```


```
Project Structure:

my_project/
├── build.zig       <- Build configuration
├── build.zig.zon   <- Dependencies
└── src/
    ├── main.zig    <- Entry point
    └── lib.zig     <- Library code

Build Process:
┌────────────┐   ┌────────────┐   ┌────────────┐
│ build.zig  │ → │ zig build  │ → │ zig-out/   │
└────────────┘   └────────────┘   └────────────┘
```


### Example: build_example

Project structure and build commands

```zig
const std = @import("std");

// Example of a library function
pub fn greet(name: []const u8) void {
    std.debug.print("Hello, {s}!\n", .{name});
}

// Example of configuration
pub const Config = struct {
    debug: bool = false,
    optimization: enum { none, speed, size } = .none,
};

pub fn main() void {
    std.debug.print("Build System Demo\n", .{});
    std.debug.print("================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Commands:\n", .{});
    std.debug.print("  zig init        - Create new project\n", .{});
    std.debug.print("  zig build       - Build project\n", .{});
    std.debug.print("  zig build run   - Build and run\n", .{});
    std.debug.print("  zig build test  - Run tests\n", .{});
    std.debug.print("\n", .{});
    greet("World");
}
```

**Output:**
```
Build System Demo
================

Commands:
  zig init        - Create new project
  zig build       - Build project
  zig build run   - Build and run
  zig build test  - Run tests

Hello, World!
```


---

## Quick Reference


### Common Commands
```
zig run file.zig     # Compile and run
zig build-exe file.zig  # Compile to executable
zig test file.zig    # Run tests
zig fmt file.zig     # Format code
zig init             # Initialize new project
zig build            # Build project
```

### Useful Built-ins
| Built-in | Description |
|----------|-------------|
| @import | Import module |
| @intCast | Cast integer types |
| @floatFromInt | Convert int to float |
| @truncate | Truncate to smaller type |
| @sizeOf | Get size of type |
| @TypeOf | Get type of expression |

### Format Specifiers
| Specifier | Description |
|-----------|-------------|
| {} | Default |
| {s} | String |
| {c} | Character |
| {d} | Decimal |
| {x} | Hexadecimal |
| {b} | Binary |
