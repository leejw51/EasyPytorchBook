// Chapter 2: Variables
// const, var, and comptime variables

const std = @import("std");

pub fn main() void {
    // Constants - cannot be changed after initialization
    const pi: f64 = 3.14159;
    const greeting = "Hello"; // Type inferred as *const [5:0]u8

    std.debug.print("pi = {d}\n", .{pi});
    std.debug.print("greeting = {s}\n", .{greeting});

    // Variables - can be changed
    var counter: i32 = 0;
    counter += 1;
    counter += 1;
    std.debug.print("counter = {d}\n", .{counter});

    // Shadowing is not allowed in Zig
    // const x = 5;
    // const x = 10; // Error: redefinition

    // Undefined initialization
    var uninitialized: i32 = undefined;
    uninitialized = 42;
    std.debug.print("now initialized = {d}\n", .{uninitialized});
}

// Compile-time constants
const BUFFER_SIZE: usize = 1024;
const MAX_CONNECTIONS: u32 = 100;

// Computed compile-time constant
const DOUBLE_BUFFER: usize = BUFFER_SIZE * 2;

pub fn comptime_example() void {
    std.debug.print("Buffer size: {d}\n", .{BUFFER_SIZE});
    std.debug.print("Double buffer: {d}\n", .{DOUBLE_BUFFER});
}

// Block expressions for complex initialization
pub fn block_example() void {
    const value = blk: {
        var temp: i32 = 10;
        temp *= 2;
        temp += 5;
        break :blk temp;
    };
    std.debug.print("Block result: {d}\n", .{value});
}
