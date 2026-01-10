// Chapter 13: Functions
// Function definitions and calling conventions

const std = @import("std");

// Basic function
fn add(a: i32, b: i32) i32 {
    return a + b;
}

// Function with no return value
fn greet(name: []const u8) void {
    std.debug.print("Hello, {s}!\n", .{name});
}

// Function returning multiple values (tuple)
fn divmod(numerator: i32, denominator: i32) struct { quotient: i32, remainder: i32 } {
    return .{
        .quotient = @divTrunc(numerator, denominator),
        .remainder = @mod(numerator, denominator),
    };
}

// Function with error union return
fn divide(a: i32, b: i32) !i32 {
    if (b == 0) return error.DivisionByZero;
    return @divTrunc(a, b);
}

pub fn main() void {
    basic_functions();
    function_pointers();
    higher_order();
    recursion();
}

fn basic_functions() void {
    std.debug.print("=== Basic Functions ===\n", .{});

    // Call simple function
    const sum = add(5, 3);
    std.debug.print("5 + 3 = {d}\n", .{sum});

    // Void function
    greet("Zig");

    // Multiple return values
    const result = divmod(17, 5);
    std.debug.print("17 / 5 = {d} remainder {d}\n", .{ result.quotient, result.remainder });

    // Destructure return value
    const dm = divmod(23, 7);
    std.debug.print("23 / 7: q={d}, r={d}\n", .{ dm.quotient, dm.remainder });
}

// Function type
const BinaryOp = *const fn (i32, i32) i32;

fn subtract(a: i32, b: i32) i32 {
    return a - b;
}

fn multiply(a: i32, b: i32) i32 {
    return a * b;
}

fn apply(op: BinaryOp, a: i32, b: i32) i32 {
    return op(a, b);
}

fn function_pointers() void {
    std.debug.print("\n=== Function Pointers ===\n", .{});

    // Store function in variable
    const op1: BinaryOp = add;
    const op2: BinaryOp = subtract;
    const op3: BinaryOp = multiply;

    std.debug.print("add(10, 3) = {d}\n", .{op1(10, 3)});
    std.debug.print("sub(10, 3) = {d}\n", .{op2(10, 3)});
    std.debug.print("mul(10, 3) = {d}\n", .{op3(10, 3)});

    // Pass function as argument
    std.debug.print("apply(add, 5, 2) = {d}\n", .{apply(add, 5, 2)});
    std.debug.print("apply(mul, 5, 2) = {d}\n", .{apply(multiply, 5, 2)});
}

// Higher-order function: map
fn map(comptime T: type, arr: []const T, f: *const fn (T) T) [arr.len]T {
    var result: [arr.len]T = undefined;
    for (arr, 0..) |item, i| {
        result[i] = f(item);
    }
    return result;
}

fn double(x: i32) i32 {
    return x * 2;
}

fn square(x: i32) i32 {
    return x * x;
}

fn higher_order() void {
    std.debug.print("\n=== Higher-Order Functions ===\n", .{});

    const numbers = [_]i32{ 1, 2, 3, 4, 5 };

    // Map with double
    std.debug.print("Original: {any}\n", .{numbers});

    // Use inline for to apply function
    var doubled: [5]i32 = undefined;
    for (numbers, 0..) |n, i| {
        doubled[i] = double(n);
    }
    std.debug.print("Doubled: {any}\n", .{doubled});

    var squared: [5]i32 = undefined;
    for (numbers, 0..) |n, i| {
        squared[i] = square(n);
    }
    std.debug.print("Squared: {any}\n", .{squared});

    // Filter (count matches)
    var count: usize = 0;
    for (numbers) |n| {
        if (n > 2) count += 1;
    }
    std.debug.print("Count > 2: {d}\n", .{count});

    // Reduce (fold)
    var sum: i32 = 0;
    for (numbers) |n| {
        sum += n;
    }
    std.debug.print("Sum: {d}\n", .{sum});
}

// Recursive function
fn factorial(n: u64) u64 {
    if (n == 0) return 1;
    return n * factorial(n - 1);
}

fn fibonacci(n: u32) u32 {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Tail-recursive factorial
fn factorial_tail(n: u64, acc: u64) u64 {
    if (n == 0) return acc;
    return factorial_tail(n - 1, n * acc);
}

fn recursion() void {
    std.debug.print("\n=== Recursion ===\n", .{});

    std.debug.print("5! = {d}\n", .{factorial(5)});
    std.debug.print("10! = {d}\n", .{factorial(10)});

    std.debug.print("fib(10) = {d}\n", .{fibonacci(10)});

    // Tail recursive version
    std.debug.print("5! (tail) = {d}\n", .{factorial_tail(5, 1)});
}

// Inline function (always inlined)
inline fn inlined_add(a: i32, b: i32) i32 {
    return a + b;
}

// Export for C interop
export fn exported_function(x: i32) i32 {
    return x * 2;
}

// External function declaration
extern "c" fn abs(x: c_int) c_int;

pub fn c_interop_example() void {
    std.debug.print("\n=== C Interop ===\n", .{});
    const result = abs(-42);
    std.debug.print("abs(-42) = {d}\n", .{result});
}
