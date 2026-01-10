// Chapter 8: Strings
// String handling in Zig

const std = @import("std");

pub fn main() void {
    string_basics();
    string_operations();
    string_formatting();
    multiline_strings();
}

fn string_basics() void {
    std.debug.print("=== String Basics ===\n", .{});

    // String literals are []const u8 (slice of bytes)
    const greeting: []const u8 = "Hello, Zig!";
    std.debug.print("greeting: {s}\n", .{greeting});
    std.debug.print("length: {d}\n", .{greeting.len});

    // Access individual bytes
    std.debug.print("First byte: {c}\n", .{greeting[0]});
    std.debug.print("As number: {d}\n", .{greeting[0]});

    // String pointer type
    const ptr_str: [*:0]const u8 = "Null-terminated";
    std.debug.print("Pointer string: {s}\n", .{ptr_str});

    // Coerce to slice
    const slice: [:0]const u8 = "Also null-terminated";
    std.debug.print("Slice string: {s}\n", .{slice});
    std.debug.print("Sentinel: {d}\n", .{slice[slice.len]}); // 0
}

fn string_operations() void {
    std.debug.print("\n=== String Operations ===\n", .{});

    const str = "Hello, World!";

    // Iteration
    std.debug.print("Characters: ", .{});
    for (str) |char| {
        std.debug.print("{c}", .{char});
    }
    std.debug.print("\n", .{});

    // Substring (slicing)
    const hello = str[0..5];
    const world = str[7..12];
    std.debug.print("hello: {s}\n", .{hello});
    std.debug.print("world: {s}\n", .{world});

    // String comparison
    const a = "apple";
    const b = "apple";
    const c = "banana";

    std.debug.print("a == b: {}\n", .{std.mem.eql(u8, a, b)});
    std.debug.print("a == c: {}\n", .{std.mem.eql(u8, a, c)});

    // Find substring
    const text = "The quick brown fox";
    if (std.mem.indexOf(u8, text, "quick")) |index| {
        std.debug.print("'quick' found at index: {d}\n", .{index});
    }

    // Check prefix/suffix
    std.debug.print("Starts with 'The': {}\n", .{std.mem.startsWith(u8, text, "The")});
    std.debug.print("Ends with 'fox': {}\n", .{std.mem.endsWith(u8, text, "fox")});
}

fn string_formatting() void {
    std.debug.print("\n=== String Formatting ===\n", .{});

    // Format specifiers
    const num: i32 = 42;
    const float: f64 = 3.14159;
    const str = "Zig";

    std.debug.print("Integer: {d}\n", .{num});
    std.debug.print("Float: {d:.2}\n", .{float}); // 2 decimal places
    std.debug.print("String: {s}\n", .{str});
    std.debug.print("Character: {c}\n", .{'Z'});

    // Number bases
    const byte: u8 = 255;
    std.debug.print("Decimal: {d}\n", .{byte});
    std.debug.print("Hex: 0x{x}\n", .{byte});
    std.debug.print("Binary: 0b{b}\n", .{byte});
    std.debug.print("Octal: 0o{o}\n", .{byte});

    // Padding and alignment
    std.debug.print("Right aligned: |{d:>10}|\n", .{42});
    std.debug.print("Left aligned:  |{d:<10}|\n", .{42});
    std.debug.print("Zero padded:   |{d:0>10}|\n", .{42});
}

fn multiline_strings() void {
    std.debug.print("\n=== Multiline Strings ===\n", .{});

    // Multiline string literal
    const poem =
        \\Roses are red,
        \\Violets are blue,
        \\Zig is awesome,
        \\And so are you!
    ;

    std.debug.print("{s}\n", .{poem});

    // Escape sequences
    const escaped = "Tab:\tNewline:\nQuote:\"Backslash:\\";
    std.debug.print("Escaped: {s}\n", .{escaped});

    // Unicode
    const unicode = "Hello, 世界! 🚀";
    std.debug.print("Unicode: {s}\n", .{unicode});
}

// Working with mutable strings
pub fn mutable_strings() void {
    std.debug.print("\n=== Mutable Strings ===\n", .{});

    // Create mutable buffer
    var buffer: [100]u8 = undefined;

    // Copy string into buffer
    const src = "Hello";
    @memcpy(buffer[0..src.len], src);

    std.debug.print("Buffer: {s}\n", .{buffer[0..src.len]});

    // Modify buffer
    buffer[0] = 'J';
    std.debug.print("Modified: {s}\n", .{buffer[0..src.len]});
}

// String concatenation at comptime
const GREETING = "Hello, ";
const NAME = "World";
const MESSAGE = GREETING ++ NAME ++ "!";

pub fn comptime_concat() void {
    std.debug.print("\n=== Comptime Concatenation ===\n", .{});
    std.debug.print("{s}\n", .{MESSAGE});
}
