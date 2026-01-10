// Chapter 1: Hello World
// Your first Zig program

const std = @import("std");

pub fn main() void {
    // Print to standard output
    std.debug.print("Hello, World!\n", .{});
}

// Example with formatted output
pub fn main2() void {
    const name = "Zig";
    const year: u32 = 2016;

    std.debug.print("Welcome to {s}!\n", .{name});
    std.debug.print("{s} was created in {d}.\n", .{ name, year });
}

// Example with multiple print statements
pub fn main3() void {
    std.debug.print("Line 1\n", .{});
    std.debug.print("Line 2\n", .{});
    std.debug.print("The answer is: {d}\n", .{42});
}
