---
title: Easy Zig
author: JW Lee
language: en
---

# Easy Zig

**A Beginner's Guide to the Zig Programming Language**

*by JW Lee*

---

# Preface

Welcome to **Easy Zig**! This book is designed for beginners who want to learn the Zig programming language from scratch.

Zig is a modern systems programming language that aims to be simple, fast, and safe. It's designed to be a better C - giving you low-level control without the complexity and pitfalls of C/C++.

## Why Zig?

- **Simple**: No hidden control flow, no hidden memory allocations
- **Fast**: Compiles to highly optimized machine code
- **Safe**: Catches bugs at compile time, not runtime
- **Portable**: Works on many platforms including embedded systems
- **C Compatible**: Can call C libraries directly

## How to Use This Book

Each chapter introduces one concept with simple examples. Read the code, run it yourself, and experiment!

To run any example:

```bash
zig run filename.zig
```

Let's begin your Zig journey!

---

# Chapter 1: Hello World

Every programming journey starts with Hello World. Let's write our first Zig program.

## Your First Program

```zig
// Import the standard library
const std = @import("std");

// Main function - the entry point of our program
pub fn main() void {
    // Print "Hello, World!" to the console
    // .{} is an empty tuple for format arguments
    std.debug.print("Hello, World!\n", .{});
}
```

**Output:**
```
Hello, World!
```

## Printing with Variables

```zig
const std = @import("std");

pub fn main() void {
    // Declare constants
    const name = "Zig";        // String literal
    const year: u32 = 2016;    // Unsigned 32-bit integer

    // {s} formats a string, {d} formats a decimal number
    std.debug.print("Welcome to {s}!\n", .{name});
    std.debug.print("{s} was created in {d}.\n", .{ name, year });
}
```

**Output:**
```
Welcome to Zig!
Zig was created in 2016.
```

## Key Points

- `const std = @import("std")` - Import the standard library
- `pub fn main() void` - Public main function that returns nothing
- `std.debug.print()` - Print to console (for debugging)
- `{s}` - Format specifier for strings
- `{d}` - Format specifier for decimal numbers
- `.{}` - Tuple containing format arguments

---

# Chapter 2: Variables

Zig has two ways to store values: constants (`const`) and variables (`var`).

## Constants vs Variables

```zig
const std = @import("std");

pub fn main() void {
    // CONSTANTS - cannot be changed after initialization
    const pi: f64 = 3.14159;       // Explicit type
    const greeting = "Hello";       // Type inferred

    std.debug.print("pi = {d}\n", .{pi});
    std.debug.print("greeting = {s}\n", .{greeting});

    // VARIABLES - can be changed
    var counter: i32 = 0;          // Start at 0
    counter += 1;                   // Now it's 1
    counter += 1;                   // Now it's 2
    std.debug.print("counter = {d}\n", .{counter});

    // UNDEFINED - declare now, assign later
    var value: i32 = undefined;    // Not initialized yet
    value = 42;                     // Now it has a value
    std.debug.print("value = {d}\n", .{value});
}
```

**Output:**
```
pi = 3.14159
greeting = Hello
counter = 2
value = 42
```

## Compile-Time Constants

```zig
const std = @import("std");

// These are computed at compile time (not runtime)
const BUFFER_SIZE: usize = 1024;
const MAX_USERS: u32 = 100;
const DOUBLE_BUFFER: usize = BUFFER_SIZE * 2;  // Computed!

pub fn main() void {
    std.debug.print("Buffer: {d} bytes\n", .{BUFFER_SIZE});
    std.debug.print("Double: {d} bytes\n", .{DOUBLE_BUFFER});
}
```

**Output:**
```
Buffer: 1024 bytes
Double: 2048 bytes
```

## Key Points

- Use `const` when the value won't change (preferred)
- Use `var` when you need to modify the value
- `undefined` means "I'll assign this later"
- Compile-time constants are computed before your program runs

---

# Chapter 3: Types

Zig has many built-in types for different purposes.

## Integer Types

```zig
const std = @import("std");

pub fn main() void {
    // Signed integers (can be negative)
    const a: i8 = -128;            // -128 to 127
    const b: i16 = -32768;         // Larger range
    const c: i32 = -2147483648;    // Even larger
    const d: i64 = -9223372036854775808;  // Biggest

    // Unsigned integers (only positive)
    const e: u8 = 255;             // 0 to 255
    const f: u16 = 65535;          // 0 to 65535
    const g: u32 = 4294967295;     // Larger
    const h: u64 = 18446744073709551615;  // Biggest

    std.debug.print("i8: {d}, u8: {d}\n", .{ a, e });
    std.debug.print("i16: {d}, u16: {d}\n", .{ b, f });
    std.debug.print("i32: {d}, u32: {d}\n", .{ c, g });
}
```

**Output:**
```
i8: -128, u8: 255
i16: -32768, u16: 65535
i32: -2147483648, u32: 4294967295
```

## Floating Point and Boolean

```zig
const std = @import("std");

pub fn main() void {
    // Floating point numbers (decimals)
    const pi: f32 = 3.14159;           // 32-bit float
    const e: f64 = 2.71828182845904;   // 64-bit float (more precise)

    std.debug.print("pi = {d}\n", .{pi});
    std.debug.print("e = {d}\n", .{e});

    // Boolean (true or false)
    const is_valid: bool = true;
    const is_empty: bool = false;

    std.debug.print("is_valid = {}\n", .{is_valid});
    std.debug.print("is_empty = {}\n", .{is_empty});
}
```

**Output:**
```
pi = 3.14159
e = 2.71828182845904
is_valid = true
is_empty = false
```

## Type Casting

```zig
const std = @import("std");

pub fn main() void {
    // Small to big (automatic, safe)
    const small: u8 = 10;
    const big: u32 = small;  // OK: u8 fits in u32

    // Big to small (explicit, you must use @intCast)
    const large: u32 = 100;
    const tiny: u8 = @intCast(large);  // You take responsibility!

    // Float to int
    const float_val: f32 = 3.7;
    const int_val: i32 = @intFromFloat(float_val);  // Becomes 3

    std.debug.print("small->big: {d}\n", .{big});
    std.debug.print("large->tiny: {d}\n", .{tiny});
    std.debug.print("float->int: {d}\n", .{int_val});
}
```

**Output:**
```
small->big: 10
large->tiny: 100
float->int: 3
```

## Key Points

- `i` prefix = signed (can be negative)
- `u` prefix = unsigned (only positive)
- Number after = bits (8, 16, 32, 64)
- Use `@intCast` for narrowing conversions
- Use `@intFromFloat` to convert float to int

---

# Chapter 4: Operators

Zig supports all the common operators you'd expect.

## Arithmetic Operators

```zig
const std = @import("std");

pub fn main() void {
    const a: i32 = 10;
    const b: i32 = 3;

    // Basic math
    std.debug.print("{d} + {d} = {d}\n", .{ a, b, a + b });   // Addition
    std.debug.print("{d} - {d} = {d}\n", .{ a, b, a - b });   // Subtraction
    std.debug.print("{d} * {d} = {d}\n", .{ a, b, a * b });   // Multiplication

    // Division and modulo need special functions
    std.debug.print("{d} / {d} = {d}\n", .{ a, b, @divTrunc(a, b) });  // 3
    std.debug.print("{d} %% {d} = {d}\n", .{ a, b, @mod(a, b) });       // 1

    // Negation
    std.debug.print("-{d} = {d}\n", .{ a, -a });  // -10
}
```

**Output:**
```
10 + 3 = 13
10 - 3 = 7
10 * 3 = 30
10 / 3 = 3
10 % 3 = 1
-10 = -10
```

## Comparison and Logical Operators

```zig
const std = @import("std");

pub fn main() void {
    const x: i32 = 5;
    const y: i32 = 10;

    // Comparison operators
    std.debug.print("{d} == {d}: {}\n", .{ x, y, x == y });  // Equal
    std.debug.print("{d} != {d}: {}\n", .{ x, y, x != y });  // Not equal
    std.debug.print("{d} < {d}: {}\n", .{ x, y, x < y });    // Less than
    std.debug.print("{d} > {d}: {}\n", .{ x, y, x > y });    // Greater than

    // Logical operators
    const t = true;
    const f = false;

    std.debug.print("{} and {}: {}\n", .{ t, f, t and f });  // false
    std.debug.print("{} or {}: {}\n", .{ t, f, t or f });    // true
    std.debug.print("not {}: {}\n", .{ t, !t });              // false
}
```

**Output:**
```
5 == 10: false
5 != 10: true
5 < 10: true
5 > 10: false
true and false: false
true or false: true
not true: false
```

## Bitwise Operators

```zig
const std = @import("std");

pub fn main() void {
    const a: u8 = 0b11001010;  // 202 in binary
    const b: u8 = 0b10101100;  // 172 in binary

    std.debug.print("a     = 0b{b:0>8}\n", .{a});      // 11001010
    std.debug.print("b     = 0b{b:0>8}\n", .{b});      // 10101100
    std.debug.print("a & b = 0b{b:0>8}\n", .{a & b});  // AND: 10001000
    std.debug.print("a | b = 0b{b:0>8}\n", .{a | b});  // OR:  11101110
    std.debug.print("a ^ b = 0b{b:0>8}\n", .{a ^ b});  // XOR: 01100110

    // Bit shifts
    const c: u8 = 0b00001111;  // 15
    std.debug.print("c << 2 = 0b{b:0>8}\n", .{c << 2});  // 00111100
    std.debug.print("c >> 2 = 0b{b:0>8}\n", .{c >> 2});  // 00000011
}
```

**Output:**
```
a     = 0b11001010
b     = 0b10101100
a & b = 0b10001000
a | b = 0b11101110
a ^ b = 0b01100110
c << 2 = 0b00111100
c >> 2 = 0b00000011
```

---

# Chapter 5: Arrays

Arrays are fixed-size collections of elements of the same type.

## Creating Arrays

```zig
const std = @import("std");

pub fn main() void {
    // Array with explicit type: [size]type
    const numbers: [5]i32 = [5]i32{ 1, 2, 3, 4, 5 };

    // Array with inferred length: [_] means "figure out the size"
    const fruits = [_][]const u8{ "apple", "banana", "cherry" };

    // Access elements by index (starts at 0)
    std.debug.print("First number: {d}\n", .{numbers[0]});
    std.debug.print("Last number: {d}\n", .{numbers[4]});
    std.debug.print("First fruit: {s}\n", .{fruits[0]});

    // Get array length
    std.debug.print("Numbers length: {d}\n", .{numbers.len});
    std.debug.print("Fruits length: {d}\n", .{fruits.len});
}
```

**Output:**
```
First number: 1
Last number: 5
First fruit: apple
Numbers length: 5
Fruits length: 3
```

## Array Initialization Patterns

```zig
const std = @import("std");

pub fn main() void {
    // Fill with same value: [_]T{value} ** count
    const zeros = [_]i32{0} ** 5;        // [0, 0, 0, 0, 0]
    const ones = [_]i32{1} ** 3;         // [1, 1, 1]

    // Repeat a pattern
    const pattern = [_]i32{ 1, 2 } ** 3;  // [1, 2, 1, 2, 1, 2]

    std.debug.print("zeros: {any}\n", .{zeros});
    std.debug.print("ones: {any}\n", .{ones});
    std.debug.print("pattern: {any}\n", .{pattern});
}
```

**Output:**
```
zeros: { 0, 0, 0, 0, 0 }
ones: { 1, 1, 1 }
pattern: { 1, 2, 1, 2, 1, 2 }
```

## Iterating Over Arrays

```zig
const std = @import("std");

pub fn main() void {
    const numbers = [_]i32{ 10, 20, 30, 40, 50 };

    // Simple iteration
    std.debug.print("Values: ", .{});
    for (numbers) |value| {
        std.debug.print("{d} ", .{value});
    }
    std.debug.print("\n", .{});

    // Iteration with index
    std.debug.print("With index:\n", .{});
    for (numbers, 0..) |value, index| {
        std.debug.print("  [{d}] = {d}\n", .{ index, value });
    }
}
```

**Output:**
```
Values: 10 20 30 40 50
With index:
  [0] = 10
  [1] = 20
  [2] = 30
  [3] = 40
  [4] = 50
```

## Modifying Arrays

```zig
const std = @import("std");

pub fn main() void {
    // Must be 'var' to modify
    var arr = [_]i32{ 1, 2, 3, 4, 5 };

    std.debug.print("Before: {any}\n", .{arr});

    // Modify single elements
    arr[0] = 100;
    arr[4] = 500;

    std.debug.print("After: {any}\n", .{arr});

    // Modify all elements using pointer capture
    for (&arr) |*item| {
        item.* *= 2;  // Double each element
    }

    std.debug.print("Doubled: {any}\n", .{arr});
}
```

**Output:**
```
Before: { 1, 2, 3, 4, 5 }
After: { 100, 2, 3, 4, 500 }
Doubled: { 200, 4, 6, 8, 1000 }
```

---

# Chapter 6: Slices

Slices are views into arrays or other memory. They're more flexible than arrays.

## Creating Slices

```zig
const std = @import("std");

pub fn main() void {
    const array = [_]i32{ 10, 20, 30, 40, 50, 60, 70 };

    // Create slice from entire array
    const all: []const i32 = &array;
    std.debug.print("All: {any}\n", .{all});

    // Slice with range [start..end)
    const middle = array[2..5];  // Elements 2, 3, 4
    std.debug.print("middle[2..5]: {any}\n", .{middle});

    // Slice from start
    const first_three = array[0..3];
    std.debug.print("first_three: {any}\n", .{first_three});

    // Slice to end
    const last_four = array[3..];
    std.debug.print("last_four: {any}\n", .{last_four});
}
```

**Output:**
```
All: { 10, 20, 30, 40, 50, 60, 70 }
middle[2..5]: { 30, 40, 50 }
first_three: { 10, 20, 30 }
last_four: { 40, 50, 60, 70 }
```

## Slices as Function Parameters

```zig
const std = @import("std");

// Function takes a slice - works with any size array!
fn sum(values: []const i32) i32 {
    var total: i32 = 0;
    for (values) |v| {
        total += v;
    }
    return total;
}

fn printSlice(data: []const u8) void {
    std.debug.print("Data: {s}\n", .{data});
}

pub fn main() void {
    // Different sized arrays
    const arr1 = [_]i32{ 1, 2, 3 };
    const arr2 = [_]i32{ 10, 20, 30, 40, 50 };

    // Same function works with both!
    std.debug.print("Sum of arr1: {d}\n", .{sum(&arr1)});
    std.debug.print("Sum of arr2: {d}\n", .{sum(&arr2)});

    // Even works with partial slices
    std.debug.print("Sum of arr2[1..4]: {d}\n", .{sum(arr2[1..4])});

    // String slice
    printSlice("Hello, Zig!");
}
```

**Output:**
```
Sum of arr1: 6
Sum of arr2: 150
Sum of arr2[1..4]: 90
Data: Hello, Zig!
```

## Modifying Through Slices

```zig
const std = @import("std");

pub fn main() void {
    var array = [_]i32{ 1, 2, 3, 4, 5 };

    // Get mutable slice
    const slice: []i32 = &array;

    std.debug.print("Before: {any}\n", .{slice});

    // Modify through slice
    slice[0] = 100;
    slice[4] = 500;

    std.debug.print("After: {any}\n", .{slice});

    // Original array is also modified!
    std.debug.print("Array: {any}\n", .{array});
}
```

**Output:**
```
Before: { 1, 2, 3, 4, 5 }
After: { 100, 2, 3, 4, 500 }
Array: { 100, 2, 3, 4, 500 }
```

---

# Chapter 7: Pointers

Pointers hold the memory address of a value.

## Basic Pointers

```zig
const std = @import("std");

pub fn main() void {
    var value: i32 = 42;

    // Get pointer to the variable
    const ptr: *i32 = &value;

    std.debug.print("value = {d}\n", .{value});
    std.debug.print("ptr.* = {d}\n", .{ptr.*});  // Dereference

    // Modify through pointer
    ptr.* = 100;
    std.debug.print("After ptr.* = 100:\n", .{});
    std.debug.print("  value = {d}\n", .{value});  // Also 100!
}
```

**Output:**
```
value = 42
ptr.* = 42
After ptr.* = 100:
  value = 100
```

## Passing by Pointer

```zig
const std = @import("std");

// This function can modify the original value
fn increment(ptr: *i32) void {
    ptr.* += 1;
}

fn double_all(values: []i32) void {
    for (values) |*v| {
        v.* *= 2;
    }
}

pub fn main() void {
    var num: i32 = 5;
    std.debug.print("Before: {d}\n", .{num});

    increment(&num);
    std.debug.print("After increment: {d}\n", .{num});

    increment(&num);
    increment(&num);
    std.debug.print("After 2 more: {d}\n", .{num});

    // Modify array elements
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    std.debug.print("Array before: {any}\n", .{arr});

    double_all(&arr);
    std.debug.print("Array after: {any}\n", .{arr});
}
```

**Output:**
```
Before: 5
After increment: 6
After 2 more: 8
Array before: { 1, 2, 3, 4, 5 }
Array after: { 2, 4, 6, 8, 10 }
```

## Const vs Mutable Pointers

```zig
const std = @import("std");

pub fn main() void {
    var mutable: i32 = 10;
    const immutable: i32 = 20;

    // Pointer to mutable - can modify
    const ptr_mut: *i32 = &mutable;
    ptr_mut.* = 15;
    std.debug.print("Modified mutable: {d}\n", .{mutable});

    // Pointer to const - cannot modify
    const ptr_const: *const i32 = &immutable;
    std.debug.print("Const value: {d}\n", .{ptr_const.*});
    // ptr_const.* = 25;  // ERROR: cannot modify const
}
```

**Output:**
```
Modified mutable: 15
Const value: 20
```

---

# Chapter 8: Strings

Strings in Zig are just arrays of bytes (`[]const u8`).

## String Basics

```zig
const std = @import("std");

pub fn main() void {
    // String literals are []const u8
    const greeting: []const u8 = "Hello, Zig!";

    std.debug.print("String: {s}\n", .{greeting});
    std.debug.print("Length: {d}\n", .{greeting.len});

    // Access individual bytes
    std.debug.print("First byte: {c}\n", .{greeting[0]});  // 'H'
    std.debug.print("As number: {d}\n", .{greeting[0]});   // 72

    // Iterate over characters
    std.debug.print("Characters: ", .{});
    for (greeting) |char| {
        std.debug.print("{c}", .{char});
    }
    std.debug.print("\n", .{});
}
```

**Output:**
```
String: Hello, Zig!
Length: 11
First byte: H
As number: 72
Characters: Hello, Zig!
```

## String Operations

```zig
const std = @import("std");

pub fn main() void {
    const str = "Hello, World!";

    // Substring (slicing)
    const hello = str[0..5];   // "Hello"
    const world = str[7..12];  // "World"

    std.debug.print("hello: {s}\n", .{hello});
    std.debug.print("world: {s}\n", .{world});

    // String comparison
    const a = "apple";
    const b = "apple";
    const c = "banana";

    std.debug.print("a == b: {}\n", .{std.mem.eql(u8, a, b)});  // true
    std.debug.print("a == c: {}\n", .{std.mem.eql(u8, a, c)});  // false

    // Find substring
    const text = "The quick brown fox";
    if (std.mem.indexOf(u8, text, "quick")) |index| {
        std.debug.print("'quick' at index: {d}\n", .{index});
    }

    // Check prefix/suffix
    std.debug.print("Starts with 'The': {}\n", .{
        std.mem.startsWith(u8, text, "The"),
    });
    std.debug.print("Ends with 'fox': {}\n", .{
        std.mem.endsWith(u8, text, "fox"),
    });
}
```

**Output:**
```
hello: Hello
world: World
a == b: true
a == c: false
'quick' at index: 4
Starts with 'The': true
Ends with 'fox': true
```

## Multiline Strings

```zig
const std = @import("std");

pub fn main() void {
    // Multiline string with \\
    const poem =
        \\Roses are red,
        \\Violets are blue,
        \\Zig is awesome,
        \\And so are you!
    ;

    std.debug.print("{s}\n", .{poem});

    // Escape sequences
    const escaped = "Tab:\tNewline:\nQuote:\"";
    std.debug.print("{s}\n", .{escaped});

    // Unicode works too
    const unicode = "Hello, World! Earth";
    std.debug.print("{s}\n", .{unicode});
}
```

**Output:**
```
Roses are red,
Violets are blue,
Zig is awesome,
And so are you!
Tab:	Newline:
Quote:"
Hello, World! Earth
```

---

# Chapter 9: Structs

Structs let you create custom types with named fields.

## Defining Structs

```zig
const std = @import("std");

// Define a struct type
const Point = struct {
    x: i32,
    y: i32,
};

// Struct with default values
const Config = struct {
    width: u32 = 800,
    height: u32 = 600,
    fullscreen: bool = false,
};

pub fn main() void {
    // Create instance with all fields
    const p1 = Point{ .x = 10, .y = 20 };
    std.debug.print("p1: ({d}, {d})\n", .{ p1.x, p1.y });

    // Create with default values
    const cfg1 = Config{};  // Uses all defaults
    std.debug.print("Default: {d}x{d}\n", .{ cfg1.width, cfg1.height });

    // Override some defaults
    const cfg2 = Config{
        .width = 1920,
        .height = 1080,
    };
    std.debug.print("Custom: {d}x{d}\n", .{ cfg2.width, cfg2.height });

    // Mutable struct
    var p2 = Point{ .x = 0, .y = 0 };
    p2.x = 100;
    p2.y = 200;
    std.debug.print("p2: ({d}, {d})\n", .{ p2.x, p2.y });
}
```

**Output:**
```
p1: (10, 20)
Default: 800x600
Custom: 1920x1080
p2: (100, 200)
```

## Struct Methods

```zig
const std = @import("std");

const Rectangle = struct {
    width: f64,
    height: f64,

    // Method: takes self as first parameter
    pub fn area(self: Rectangle) f64 {
        return self.width * self.height;
    }

    // Method with pointer: can modify self
    pub fn scale(self: *Rectangle, factor: f64) void {
        self.width *= factor;
        self.height *= factor;
    }

    // Associated function: no self parameter
    pub fn square(size: f64) Rectangle {
        return Rectangle{ .width = size, .height = size };
    }
};

pub fn main() void {
    var rect = Rectangle{ .width = 10.0, .height = 5.0 };

    std.debug.print("Size: {d} x {d}\n", .{ rect.width, rect.height });
    std.debug.print("Area: {d}\n", .{rect.area()});

    // Scale modifies the rectangle
    rect.scale(2.0);
    std.debug.print("After scale(2): {d} x {d}\n", .{ rect.width, rect.height });

    // Create a square using associated function
    const sq = Rectangle.square(7.0);
    std.debug.print("Square area: {d}\n", .{sq.area()});
}
```

**Output:**
```
Size: 10 x 5
Area: 50
After scale(2): 20 x 10
Square area: 49
```

## Nested Structs

```zig
const std = @import("std");

const Person = struct {
    name: []const u8,
    age: u32,
    address: Address,

    const Address = struct {
        street: []const u8,
        city: []const u8,
    };

    pub fn describe(self: Person) void {
        std.debug.print("{s}, age {d}\n", .{ self.name, self.age });
        std.debug.print("Lives at: {s}, {s}\n", .{
            self.address.street,
            self.address.city,
        });
    }
};

pub fn main() void {
    const john = Person{
        .name = "John Doe",
        .age = 30,
        .address = .{
            .street = "123 Main St",
            .city = "Springfield",
        },
    };

    john.describe();
}
```

**Output:**
```
John Doe, age 30
Lives at: 123 Main St, Springfield
```

---

# Chapter 10: Enums

Enums define a type with a fixed set of named values.

## Basic Enums

```zig
const std = @import("std");

const Color = enum {
    red,
    green,
    blue,
    yellow,
};

pub fn main() void {
    // Declare enum variable
    const favorite: Color = .blue;
    var current: Color = .red;

    std.debug.print("Favorite: {}\n", .{favorite});
    std.debug.print("Current: {}\n", .{current});

    // Change value
    current = .green;
    std.debug.print("Changed to: {}\n", .{current});

    // Compare
    if (favorite == .blue) {
        std.debug.print("Blue is the favorite!\n", .{});
    }
}
```

**Output:**
```
Favorite: Color.blue
Current: Color.red
Changed to: Color.green
Blue is the favorite!
```

## Switch on Enums

```zig
const std = @import("std");

const Direction = enum {
    north,
    south,
    east,
    west,
};

fn describe(dir: Direction) []const u8 {
    // Switch must handle ALL cases
    return switch (dir) {
        .north => "Going up",
        .south => "Going down",
        .east => "Going right",
        .west => "Going left",
    };
}

pub fn main() void {
    const directions = [_]Direction{ .north, .east, .south, .west };

    for (directions) |dir| {
        std.debug.print("{}: {s}\n", .{ dir, describe(dir) });
    }
}
```

**Output:**
```
Direction.north: Going up
Direction.east: Going right
Direction.south: Going down
Direction.west: Going left
```

## Enum with Methods

```zig
const std = @import("std");

const Direction = enum {
    north,
    south,
    east,
    west,

    pub fn opposite(self: Direction) Direction {
        return switch (self) {
            .north => .south,
            .south => .north,
            .east => .west,
            .west => .east,
        };
    }

    pub fn toVector(self: Direction) struct { x: i32, y: i32 } {
        return switch (self) {
            .north => .{ .x = 0, .y = -1 },
            .south => .{ .x = 0, .y = 1 },
            .east => .{ .x = 1, .y = 0 },
            .west => .{ .x = -1, .y = 0 },
        };
    }
};

pub fn main() void {
    const dir: Direction = .north;

    std.debug.print("Direction: {}\n", .{dir});
    std.debug.print("Opposite: {}\n", .{dir.opposite()});

    const vec = dir.toVector();
    std.debug.print("Vector: ({d}, {d})\n", .{ vec.x, vec.y });
}
```

**Output:**
```
Direction: Direction.north
Opposite: Direction.south
Vector: (0, -1)
```

---

# Chapter 11: Unions

Unions can hold one of several types, but only one at a time.

## Tagged Unions

```zig
const std = @import("std");

// Tagged union - knows which type is active
const Number = union(enum) {
    int: i64,
    float: f64,
    invalid: void,
};

pub fn main() void {
    // Create different variants
    const a: Number = .{ .int = 42 };
    const b: Number = .{ .float = 3.14 };
    const c: Number = .invalid;

    printNumber(a);
    printNumber(b);
    printNumber(c);
}

fn printNumber(n: Number) void {
    switch (n) {
        .int => |value| std.debug.print("Integer: {d}\n", .{value}),
        .float => |value| std.debug.print("Float: {d}\n", .{value}),
        .invalid => std.debug.print("Invalid\n", .{}),
    }
}
```

**Output:**
```
Integer: 42
Float: 3.14
Invalid
```

## Union with Methods

```zig
const std = @import("std");

const Value = union(enum) {
    int: i32,
    float: f64,
    text: []const u8,

    pub fn describe(self: Value) void {
        switch (self) {
            .int => |n| std.debug.print("Integer value: {d}\n", .{n}),
            .float => |f| std.debug.print("Float value: {d:.2}\n", .{f}),
            .text => |s| std.debug.print("Text value: {s}\n", .{s}),
        }
    }

    pub fn isNumeric(self: Value) bool {
        return switch (self) {
            .int, .float => true,
            .text => false,
        };
    }
};

pub fn main() void {
    const values = [_]Value{
        .{ .int = 42 },
        .{ .float = 3.14159 },
        .{ .text = "hello" },
    };

    for (values) |v| {
        v.describe();
        std.debug.print("  Is numeric: {}\n", .{v.isNumeric()});
    }
}
```

**Output:**
```
Integer value: 42
  Is numeric: true
Float value: 3.14
  Is numeric: true
Text value: hello
  Is numeric: false
```

---

# Chapter 12: Control Flow

Zig provides standard control flow: if, switch, while, and for.

## If Expressions

```zig
const std = @import("std");

pub fn main() void {
    const x: i32 = 42;

    // Basic if
    if (x > 0) {
        std.debug.print("x is positive\n", .{});
    }

    // If-else
    if (x % 2 == 0) {
        std.debug.print("x is even\n", .{});
    } else {
        std.debug.print("x is odd\n", .{});
    }

    // If-else chain
    const score: u32 = 85;
    if (score >= 90) {
        std.debug.print("Grade: A\n", .{});
    } else if (score >= 80) {
        std.debug.print("Grade: B\n", .{});
    } else if (score >= 70) {
        std.debug.print("Grade: C\n", .{});
    } else {
        std.debug.print("Grade: F\n", .{});
    }

    // If as expression (like ternary operator)
    const abs_x = if (x >= 0) x else -x;
    std.debug.print("Absolute value: {d}\n", .{abs_x});

    const message = if (x > 0) "positive" else "non-positive";
    std.debug.print("x is {s}\n", .{message});
}
```

**Output:**
```
x is positive
x is even
Grade: B
Absolute value: 42
x is positive
```

## Switch Statements

```zig
const std = @import("std");

pub fn main() void {
    // Switch on integer
    const day: u8 = 3;
    const name = switch (day) {
        1 => "Monday",
        2 => "Tuesday",
        3 => "Wednesday",
        4 => "Thursday",
        5 => "Friday",
        6, 7 => "Weekend",  // Multiple values
        else => "Invalid",
    };
    std.debug.print("Day {d} is {s}\n", .{ day, name });

    // Range in switch
    const age: u32 = 25;
    const category = switch (age) {
        0...12 => "Child",
        13...19 => "Teenager",
        20...64 => "Adult",
        else => "Senior",
    };
    std.debug.print("Age {d}: {s}\n", .{ age, category });
}
```

**Output:**
```
Day 3 is Wednesday
Age 25: Adult
```

## While Loops

```zig
const std = @import("std");

pub fn main() void {
    // Basic while
    var i: u32 = 0;
    std.debug.print("Count: ", .{});
    while (i < 5) {
        std.debug.print("{d} ", .{i});
        i += 1;
    }
    std.debug.print("\n", .{});

    // While with continue expression
    var j: u32 = 0;
    std.debug.print("Evens: ", .{});
    while (j < 10) : (j += 1) {
        if (j % 2 != 0) continue;  // Skip odd numbers
        std.debug.print("{d} ", .{j});
    }
    std.debug.print("\n", .{});

    // Break with value
    var n: u32 = 0;
    const result = while (n < 100) : (n += 1) {
        if (n * n > 50) break n;  // Return n when found
    } else 0;
    std.debug.print("First n where n*n > 50: {d}\n", .{result});
}
```

**Output:**
```
Count: 0 1 2 3 4
Evens: 0 2 4 6 8
First n where n*n > 50: 8
```

## For Loops

```zig
const std = @import("std");

pub fn main() void {
    const numbers = [_]i32{ 10, 20, 30, 40, 50 };

    // Simple iteration
    std.debug.print("Numbers: ", .{});
    for (numbers) |n| {
        std.debug.print("{d} ", .{n});
    }
    std.debug.print("\n", .{});

    // With index
    std.debug.print("Indexed:\n", .{});
    for (numbers, 0..) |n, i| {
        std.debug.print("  [{d}] = {d}\n", .{ i, n });
    }

    // Range iteration
    std.debug.print("Range 0..5: ", .{});
    for (0..5) |i| {
        std.debug.print("{d} ", .{i});
    }
    std.debug.print("\n", .{});

    // Multiple arrays
    const a = [_]i32{ 1, 2, 3 };
    const b = [_]i32{ 10, 20, 30 };
    std.debug.print("Pairs: ", .{});
    for (a, b) |x, y| {
        std.debug.print("({d},{d}) ", .{ x, y });
    }
    std.debug.print("\n", .{});
}
```

**Output:**
```
Numbers: 10 20 30 40 50
Indexed:
  [0] = 10
  [1] = 20
  [2] = 30
  [3] = 40
  [4] = 50
Range 0..5: 0 1 2 3 4
Pairs: (1,10) (2,20) (3,30)
```

---

# Chapter 13: Functions

Functions are reusable blocks of code.

## Basic Functions

```zig
const std = @import("std");

// Simple function
fn add(a: i32, b: i32) i32 {
    return a + b;
}

// Function with no return value
fn greet(name: []const u8) void {
    std.debug.print("Hello, {s}!\n", .{name});
}

// Function returning a struct (multiple values)
fn divmod(num: i32, den: i32) struct { q: i32, r: i32 } {
    return .{
        .q = @divTrunc(num, den),
        .r = @mod(num, den),
    };
}

pub fn main() void {
    // Call functions
    const sum = add(5, 3);
    std.debug.print("5 + 3 = {d}\n", .{sum});

    greet("Zig");

    const result = divmod(17, 5);
    std.debug.print("17 / 5 = {d} remainder {d}\n", .{ result.q, result.r });
}
```

**Output:**
```
5 + 3 = 8
Hello, Zig!
17 / 5 = 3 remainder 2
```

## Function Pointers

```zig
const std = @import("std");

// Define function type
const BinaryOp = *const fn (i32, i32) i32;

fn add(a: i32, b: i32) i32 {
    return a + b;
}

fn multiply(a: i32, b: i32) i32 {
    return a * b;
}

// Function that takes a function as parameter
fn apply(op: BinaryOp, a: i32, b: i32) i32 {
    return op(a, b);
}

pub fn main() void {
    // Store function in variable
    const op1: BinaryOp = add;
    const op2: BinaryOp = multiply;

    std.debug.print("add(10, 3) = {d}\n", .{op1(10, 3)});
    std.debug.print("multiply(10, 3) = {d}\n", .{op2(10, 3)});

    // Pass function as argument
    std.debug.print("apply(add, 5, 2) = {d}\n", .{apply(add, 5, 2)});
    std.debug.print("apply(multiply, 5, 2) = {d}\n", .{apply(multiply, 5, 2)});
}
```

**Output:**
```
add(10, 3) = 13
multiply(10, 3) = 30
apply(add, 5, 2) = 7
apply(multiply, 5, 2) = 10
```

## Recursion

```zig
const std = @import("std");

fn factorial(n: u64) u64 {
    if (n == 0) return 1;
    return n * factorial(n - 1);
}

fn fibonacci(n: u32) u32 {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

pub fn main() void {
    std.debug.print("5! = {d}\n", .{factorial(5)});
    std.debug.print("10! = {d}\n", .{factorial(10)});

    std.debug.print("Fibonacci sequence: ", .{});
    for (0..10) |i| {
        std.debug.print("{d} ", .{fibonacci(@intCast(i))});
    }
    std.debug.print("\n", .{});
}
```

**Output:**
```
5! = 120
10! = 3628800
Fibonacci sequence: 0 1 1 2 3 5 8 13 21 34
```

---

# Chapter 14: Error Handling

Zig has built-in error handling with error unions.

## Error Unions

```zig
const std = @import("std");

// Define custom errors
const MathError = error{
    DivisionByZero,
    NegativeNumber,
};

// Function that can fail
fn divide(a: i32, b: i32) MathError!i32 {
    if (b == 0) return error.DivisionByZero;
    return @divTrunc(a, b);
}

fn sqrt(x: i32) MathError!i32 {
    if (x < 0) return error.NegativeNumber;
    var result: i32 = 0;
    while (result * result <= x) : (result += 1) {}
    return result - 1;
}

pub fn main() void {
    // Check result with if
    if (divide(10, 2)) |value| {
        std.debug.print("10 / 2 = {d}\n", .{value});
    } else |err| {
        std.debug.print("Error: {}\n", .{err});
    }

    if (divide(10, 0)) |value| {
        std.debug.print("10 / 0 = {d}\n", .{value});
    } else |err| {
        std.debug.print("10 / 0 failed: {}\n", .{err});
    }
}
```

**Output:**
```
10 / 2 = 5
10 / 0 failed: error.DivisionByZero
```

## Try and Catch

```zig
const std = @import("std");

const MathError = error{
    DivisionByZero,
};

fn divide(a: i32, b: i32) MathError!i32 {
    if (b == 0) return error.DivisionByZero;
    return @divTrunc(a, b);
}

pub fn main() void {
    // catch with default value
    const result1 = divide(10, 0) catch 0;
    std.debug.print("With default: {d}\n", .{result1});

    // catch with error handling
    const result2 = divide(10, 0) catch |err| blk: {
        std.debug.print("Caught: {}\n", .{err});
        break :blk -1;
    };
    std.debug.print("Result: {d}\n", .{result2});

    // catch unreachable (assert no error)
    const safe = divide(20, 4) catch unreachable;
    std.debug.print("Safe result: {d}\n", .{safe});
}
```

**Output:**
```
With default: 0
Caught: error.DivisionByZero
Result: -1
Safe result: 5
```

## Error Propagation with try

```zig
const std = @import("std");

const MathError = error{
    DivisionByZero,
    NegativeNumber,
};

fn divide(a: i32, b: i32) MathError!i32 {
    if (b == 0) return error.DivisionByZero;
    return @divTrunc(a, b);
}

fn sqrt(x: i32) MathError!i32 {
    if (x < 0) return error.NegativeNumber;
    var r: i32 = 0;
    while (r * r <= x) : (r += 1) {}
    return r - 1;
}

// try propagates errors automatically
fn calculate(a: i32, b: i32) MathError!i32 {
    const quotient = try divide(a, b);  // Returns error if fails
    const root = try sqrt(quotient);     // Returns error if fails
    return root;
}

pub fn main() void {
    // Success
    if (calculate(100, 4)) |result| {
        std.debug.print("calculate(100, 4) = {d}\n", .{result});
    } else |err| {
        std.debug.print("Error: {}\n", .{err});
    }

    // Division by zero
    if (calculate(100, 0)) |result| {
        std.debug.print("Result: {d}\n", .{result});
    } else |err| {
        std.debug.print("calculate(100, 0): {}\n", .{err});
    }

    // Negative result
    if (calculate(-25, 1)) |result| {
        std.debug.print("Result: {d}\n", .{result});
    } else |err| {
        std.debug.print("calculate(-25, 1): {}\n", .{err});
    }
}
```

**Output:**
```
calculate(100, 4) = 5
calculate(100, 0): error.DivisionByZero
calculate(-25, 1): error.NegativeNumber
```

---

# Chapter 15: Optionals

Optionals represent values that might not exist.

## Basic Optionals

```zig
const std = @import("std");

pub fn main() void {
    // Optional type: ?T
    var maybe: ?i32 = 42;
    std.debug.print("maybe = {?d}\n", .{maybe});

    // Set to null
    maybe = null;
    std.debug.print("maybe (null) = {?d}\n", .{maybe});

    // Check if has value
    const some: ?[]const u8 = "Hello";
    const none: ?[]const u8 = null;

    if (some != null) {
        std.debug.print("some is not null\n", .{});
    }
    if (none == null) {
        std.debug.print("none is null\n", .{});
    }
}
```

**Output:**
```
maybe = 42
maybe (null) = null
some is not null
none is null
```

## Unwrapping Optionals

```zig
const std = @import("std");

pub fn main() void {
    const maybe: ?i32 = 42;
    const nothing: ?i32 = null;

    // If-unwrap (payload capture)
    if (maybe) |value| {
        std.debug.print("Value is: {d}\n", .{value});
    }

    if (nothing) |value| {
        std.debug.print("Value: {d}\n", .{value});
    } else {
        std.debug.print("nothing is null\n", .{});
    }

    // orelse - provide default
    const val1 = maybe orelse 0;
    const val2 = nothing orelse -1;
    std.debug.print("val1={d}, val2={d}\n", .{ val1, val2 });

    // .? operator (assert not null)
    const guaranteed = maybe.?;  // Crashes if null!
    std.debug.print("guaranteed = {d}\n", .{guaranteed});
}
```

**Output:**
```
Value is: 42
nothing is null
val1=42, val2=-1
guaranteed = 42
```

## Optional in Functions

```zig
const std = @import("std");

// Return optional when value might not exist
fn find(haystack: []const u8, needle: u8) ?usize {
    for (haystack, 0..) |byte, index| {
        if (byte == needle) return index;
    }
    return null;  // Not found
}

fn get(arr: []const i32, index: usize) ?i32 {
    if (index >= arr.len) return null;
    return arr[index];
}

pub fn main() void {
    const text = "Hello, World!";

    if (find(text, 'W')) |index| {
        std.debug.print("'W' found at index {d}\n", .{index});
    }

    if (find(text, 'Z')) |index| {
        std.debug.print("'Z' at {d}\n", .{index});
    } else {
        std.debug.print("'Z' not found\n", .{});
    }

    const numbers = [_]i32{ 10, 20, 30 };

    std.debug.print("numbers[1] = {?d}\n", .{get(&numbers, 1)});
    std.debug.print("numbers[10] = {?d}\n", .{get(&numbers, 10)});
}
```

**Output:**
```
'W' found at index 7
'Z' not found
numbers[1] = 20
numbers[10] = null
```

---

# Chapter 16: Comptime

Comptime lets you run code at compile time.

## Compile-Time Constants

```zig
const std = @import("std");

// These are computed at compile time
const KILOBYTE = 1024;
const MEGABYTE = KILOBYTE * 1024;
const GIGABYTE = MEGABYTE * 1024;

// Function executed at compile time
fn factorial(n: u64) u64 {
    if (n == 0) return 1;
    return n * factorial(n - 1);
}

// Pre-computed values
const FACT_10 = factorial(10);
const FACT_5 = factorial(5);

pub fn main() void {
    std.debug.print("1 KB = {d} bytes\n", .{KILOBYTE});
    std.debug.print("1 MB = {d} bytes\n", .{MEGABYTE});
    std.debug.print("1 GB = {d} bytes\n", .{GIGABYTE});

    std.debug.print("5! = {d}\n", .{FACT_5});
    std.debug.print("10! = {d}\n", .{FACT_10});
}
```

**Output:**
```
1 KB = 1024 bytes
1 MB = 1048576 bytes
1 GB = 1073741824 bytes
5! = 120
10! = 3628800
```

## Comptime Blocks

```zig
const std = @import("std");

pub fn main() void {
    // Compute at compile time
    const sum = comptime blk: {
        var total: u32 = 0;
        for (1..11) |i| {
            total += @as(u32, @intCast(i));
        }
        break :blk total;
    };
    std.debug.print("Sum 1-10: {d}\n", .{sum});

    // Generate array at compile time
    const squares = comptime blk: {
        var arr: [10]i32 = undefined;
        for (0..10) |i| {
            arr[i] = @as(i32, @intCast(i * i));
        }
        break :blk arr;
    };
    std.debug.print("Squares: {any}\n", .{squares});
}
```

**Output:**
```
Sum 1-10: 55
Squares: { 0, 1, 4, 9, 16, 25, 36, 49, 64, 81 }
```

## Type Reflection

```zig
const std = @import("std");

pub fn main() void {
    // Get type information
    const T = i32;
    std.debug.print("Type: {s}\n", .{@typeName(T)});
    std.debug.print("Size: {d} bytes\n", .{@sizeOf(T)});
    std.debug.print("Alignment: {d}\n", .{@alignOf(T)});

    // Struct field introspection
    const Point = struct { x: f32, y: f32, z: f32 };

    std.debug.print("\nPoint fields:\n", .{});
    inline for (std.meta.fields(Point)) |field| {
        std.debug.print("  {s}: {s}\n", .{ field.name, @typeName(field.type) });
    }
}
```

**Output:**
```
Type: i32
Size: 4 bytes
Alignment: 4

Point fields:
  x: f32
  y: f32
  z: f32
```

---

# Chapter 17: Generics

Generics let you write code that works with any type.

## Generic Functions

```zig
const std = @import("std");

// Generic max function
fn max(comptime T: type, a: T, b: T) T {
    return if (a > b) a else b;
}

fn min(comptime T: type, a: T, b: T) T {
    return if (a < b) a else b;
}

fn swap(comptime T: type, a: *T, b: *T) void {
    const temp = a.*;
    a.* = b.*;
    b.* = temp;
}

pub fn main() void {
    // Works with different types
    std.debug.print("max(i32, 5, 10) = {d}\n", .{max(i32, 5, 10)});
    std.debug.print("max(f64, 3.14, 2.71) = {d}\n", .{max(f64, 3.14, 2.71)});
    std.debug.print("min(i32, 5, 10) = {d}\n", .{min(i32, 5, 10)});

    var x: i32 = 100;
    var y: i32 = 200;
    std.debug.print("Before swap: x={d}, y={d}\n", .{ x, y });
    swap(i32, &x, &y);
    std.debug.print("After swap: x={d}, y={d}\n", .{ x, y });
}
```

**Output:**
```
max(i32, 5, 10) = 10
max(f64, 3.14, 2.71) = 3.14
min(i32, 5, 10) = 5
Before swap: x=100, y=200
After swap: x=200, y=100
```

## Generic Structs

```zig
const std = @import("std");

// Generic Stack
fn Stack(comptime T: type) type {
    return struct {
        const Self = @This();
        items: [100]T = undefined,
        count: usize = 0,

        pub fn push(self: *Self, item: T) void {
            self.items[self.count] = item;
            self.count += 1;
        }

        pub fn pop(self: *Self) ?T {
            if (self.count == 0) return null;
            self.count -= 1;
            return self.items[self.count];
        }

        pub fn len(self: *const Self) usize {
            return self.count;
        }
    };
}

pub fn main() void {
    // Integer stack
    var int_stack = Stack(i32){};

    int_stack.push(10);
    int_stack.push(20);
    int_stack.push(30);

    std.debug.print("Stack size: {d}\n", .{int_stack.len()});

    while (int_stack.pop()) |value| {
        std.debug.print("Popped: {d}\n", .{value});
    }
}
```

**Output:**
```
Stack size: 3
Popped: 30
Popped: 20
Popped: 10
```

## Generic Algorithms

```zig
const std = @import("std");

fn linearSearch(comptime T: type, arr: []const T, target: T) ?usize {
    for (arr, 0..) |item, i| {
        if (item == target) return i;
    }
    return null;
}

fn reverse(comptime T: type, items: []T) void {
    var left: usize = 0;
    var right: usize = items.len;
    while (left < right) {
        right -= 1;
        const temp = items[left];
        items[left] = items[right];
        items[right] = temp;
        left += 1;
    }
}

pub fn main() void {
    const nums = [_]i32{ 10, 20, 30, 40, 50 };
    if (linearSearch(i32, &nums, 30)) |idx| {
        std.debug.print("Found 30 at index {d}\n", .{idx});
    }

    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    std.debug.print("Before: {any}\n", .{arr});
    reverse(i32, &arr);
    std.debug.print("After: {any}\n", .{arr});
}
```

**Output:**
```
Found 30 at index 2
Before: { 1, 2, 3, 4, 5 }
After: { 5, 4, 3, 2, 1 }
```

---

# Chapter 18: Memory Management

Zig gives you control over memory allocation.

## Stack vs Heap

```zig
const std = @import("std");

pub fn main() !void {
    // STACK MEMORY - automatic, fast, limited size
    var x: i32 = 42;
    var arr: [10]i32 = undefined;

    for (&arr, 0..) |*item, i| {
        item.* = @as(i32, @intCast(i * 2));
    }

    std.debug.print("Stack variable x = {d}\n", .{x});
    std.debug.print("Stack array: {any}\n", .{arr});
}
```

**Output:**
```
Stack variable x = 42
Stack array: { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18 }
```

## Heap Allocation

```zig
const std = @import("std");

pub fn main() !void {
    // Create an allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();  // Cleanup when done
    const allocator = gpa.allocator();

    // Allocate single value
    const ptr = try allocator.create(i32);
    defer allocator.destroy(ptr);  // Free when done
    ptr.* = 100;
    std.debug.print("Allocated value: {d}\n", .{ptr.*});

    // Allocate array/slice
    const slice = try allocator.alloc(i32, 5);
    defer allocator.free(slice);  // Free when done

    for (slice, 0..) |*item, i| {
        item.* = @as(i32, @intCast((i + 1) * 10));
    }
    std.debug.print("Allocated slice: {any}\n", .{slice});
}
```

**Output:**
```
Allocated value: 100
Allocated slice: { 10, 20, 30, 40, 50 }
```

## ArrayList

```zig
const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Dynamic array that grows as needed
    var list = std.ArrayList(i32).init(allocator);
    defer list.deinit();

    try list.append(10);
    try list.append(20);
    try list.append(30);
    try list.appendSlice(&[_]i32{ 40, 50 });

    std.debug.print("List: {any}\n", .{list.items});
    std.debug.print("Length: {d}\n", .{list.items.len});

    _ = list.pop();
    std.debug.print("After pop: {any}\n", .{list.items});
}
```

**Output:**
```
List: { 10, 20, 30, 40, 50 }
Length: 5
After pop: { 10, 20, 30, 40 }
```

## Arena Allocator

```zig
const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    // Arena - frees everything at once
    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();  // Frees ALL allocations!

    const allocator = arena.allocator();

    // Make many allocations
    const a = try allocator.create(i32);
    const b = try allocator.create(i32);
    const c = try allocator.alloc(i32, 10);

    a.* = 1;
    b.* = 2;
    @memset(c, 42);

    std.debug.print("a={d}, b={d}\n", .{ a.*, b.* });
    std.debug.print("c={any}\n", .{c});

    // No need to free individually - arena.deinit() frees all!
}
```

**Output:**
```
a=1, b=2
c={ 42, 42, 42, 42, 42, 42, 42, 42, 42, 42 }
```

---

# Chapter 19: Testing

Zig has built-in testing support.

## Writing Tests

```zig
const std = @import("std");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

// Function to test
fn add(a: i32, b: i32) i32 {
    return a + b;
}

fn factorial(n: u32) u32 {
    if (n == 0) return 1;
    return n * factorial(n - 1);
}

// Test blocks
test "basic addition" {
    try expectEqual(@as(i32, 5), add(2, 3));
}

test "addition with negatives" {
    try expectEqual(@as(i32, -1), add(2, -3));
    try expectEqual(@as(i32, 0), add(5, -5));
}

test "factorial" {
    try expectEqual(@as(u32, 1), factorial(0));
    try expectEqual(@as(u32, 1), factorial(1));
    try expectEqual(@as(u32, 120), factorial(5));
}

test "expect examples" {
    try expect(add(1, 1) == 2);
    try expect(10 > 5);
    try expect(true);
}

// Run with: zig test filename.zig
pub fn main() void {
    std.debug.print("Run tests with: zig test 19_testing.zig\n", .{});
}
```

**Running tests:**
```bash
$ zig test 19_testing.zig
All 4 tests passed!
```

## Testing Errors

```zig
const std = @import("std");
const expectError = std.testing.expectError;

fn divide(a: i32, b: i32) !i32 {
    if (b == 0) return error.DivisionByZero;
    return @divTrunc(a, b);
}

test "division by zero returns error" {
    try expectError(error.DivisionByZero, divide(10, 0));
}

test "successful division" {
    const result = try divide(10, 2);
    try std.testing.expectEqual(@as(i32, 5), result);
}
```

## Testing with Allocator

```zig
const std = @import("std");

test "ArrayList operations" {
    // Use testing allocator - detects memory leaks!
    const allocator = std.testing.allocator;

    var list = std.ArrayList(i32).init(allocator);
    defer list.deinit();

    try list.append(1);
    try list.append(2);
    try list.append(3);

    try std.testing.expectEqual(@as(usize, 3), list.items.len);
    try std.testing.expectEqual(@as(i32, 1), list.items[0]);
}
```

---

# Chapter 20: C Interoperability

Zig can easily call C code and be called from C.

## Calling C Functions

```zig
const std = @import("std");
const c = @cImport({
    @cInclude("stdio.h");
    @cInclude("stdlib.h");
    @cInclude("math.h");
});

pub fn main() void {
    // C math functions
    const x: c.double = 2.0;

    const sqrt_x = c.sqrt(x);
    const pow_x = c.pow(x, 3.0);

    std.debug.print("sqrt({d}) = {d}\n", .{ x, sqrt_x });
    std.debug.print("pow({d}, 3) = {d}\n", .{ x, pow_x });

    // C printf
    _ = c.printf("Hello from C printf!\n");

    // C abs
    const neg: c_int = -42;
    std.debug.print("abs({d}) = {d}\n", .{ neg, c.abs(neg) });
}
```

**Output:**
```
sqrt(2) = 1.41421356...
pow(2, 3) = 8
Hello from C printf!
abs(-42) = 42
```

## C String Handling

```zig
const std = @import("std");
const c = @cImport({
    @cInclude("string.h");
});

pub fn main() void {
    // Zig string to C string
    const zig_str: [:0]const u8 = "Hello from Zig!";
    const c_str: [*c]const u8 = zig_str.ptr;

    // Use C string functions
    const len = c.strlen(c_str);
    std.debug.print("String: {s}\n", .{zig_str});
    std.debug.print("Length (strlen): {d}\n", .{len});

    // String comparison
    const a: [*c]const u8 = "apple";
    const b: [*c]const u8 = "banana";
    const cmp = c.strcmp(a, b);
    std.debug.print("strcmp(apple, banana) = {d}\n", .{cmp});
}
```

**Output:**
```
String: Hello from Zig!
Length (strlen): 15
strcmp(apple, banana) = -1
```

## Exporting Functions to C

```zig
const std = @import("std");

// Export functions that C can call
export fn zig_add(a: c_int, b: c_int) c_int {
    return a + b;
}

export fn zig_factorial(n: c_uint) c_uint {
    if (n <= 1) return 1;
    return n * zig_factorial(n - 1);
}

// C-compatible struct
const CPoint = extern struct {
    x: c_int,
    y: c_int,
};

export fn create_point(x: c_int, y: c_int) CPoint {
    return CPoint{ .x = x, .y = y };
}

pub fn main() void {
    // Test our exported functions
    std.debug.print("zig_add(5, 3) = {d}\n", .{zig_add(5, 3)});
    std.debug.print("zig_factorial(5) = {d}\n", .{zig_factorial(5)});

    const p = create_point(10, 20);
    std.debug.print("Point: ({d}, {d})\n", .{ p.x, p.y });
}
```

**Output:**
```
zig_add(5, 3) = 8
zig_factorial(5) = 120
Point: (10, 20)
```

---

# Appendix A: Zig Cheat Sheet

## Types

| Type | Description |
|------|-------------|
| `i8, i16, i32, i64` | Signed integers |
| `u8, u16, u32, u64` | Unsigned integers |
| `f32, f64` | Floating point |
| `bool` | Boolean (true/false) |
| `[]T` | Slice of T |
| `[N]T` | Array of N elements |
| `*T` | Pointer to T |
| `?T` | Optional T |
| `!T` | Error union |

## Common Operations

```zig
// Variables
const x: i32 = 42;     // Immutable
var y: i32 = 0;        // Mutable

// Arrays
const arr = [_]i32{ 1, 2, 3 };
const first = arr[0];
const len = arr.len;

// Slices
const slice = arr[1..3];

// Optionals
const maybe: ?i32 = 42;
const value = maybe orelse 0;
if (maybe) |v| { ... }

// Errors
const result = try riskyFunction();
const safe = riskyFunction() catch 0;

// Loops
for (arr) |item| { ... }
for (arr, 0..) |item, i| { ... }
while (condition) { ... }
```

## Build Commands

```bash
zig run file.zig       # Compile and run
zig build-exe file.zig # Compile only
zig test file.zig      # Run tests
```

---

# Appendix B: Resources

## Official Resources

- **Zig Website**: https://ziglang.org
- **Documentation**: https://ziglang.org/documentation/master/
- **Standard Library**: https://ziglang.org/documentation/master/std/

## Community

- **Zig Discord**: https://discord.gg/ziglang
- **Reddit**: https://reddit.com/r/zig
- **GitHub**: https://github.com/ziglang/zig

---

# About the Author

**JW Lee** is a software developer passionate about systems programming and making complex topics accessible to beginners. This book represents his effort to create the simplest possible introduction to the Zig programming language.

---

*Easy Zig - A Beginner's Guide to the Zig Programming Language*

*Copyright 2024 JW Lee*

*All examples in this book are provided under the MIT License.*
