// Chapter 3: Types
// Primitive types in Zig

const std = @import("std");

pub fn main() void {
    // Integer types
    const a: i8 = -128; // 8-bit signed: -128 to 127
    const b: u8 = 255; // 8-bit unsigned: 0 to 255
    const c: i16 = -32768; // 16-bit signed
    const d: u16 = 65535; // 16-bit unsigned
    const e: i32 = -2147483648; // 32-bit signed
    const f: u32 = 4294967295; // 32-bit unsigned
    const g: i64 = -9223372036854775808; // 64-bit signed
    const h: u64 = 18446744073709551615; // 64-bit unsigned

    std.debug.print("i8: {d}, u8: {d}\n", .{ a, b });
    std.debug.print("i16: {d}, u16: {d}\n", .{ c, d });
    std.debug.print("i32: {d}, u32: {d}\n", .{ e, f });
    std.debug.print("i64: {d}, u64: {d}\n", .{ g, h });

    // Floating point types
    const pi: f32 = 3.14159; // 32-bit float
    const e_value: f64 = 2.71828182845904523536; // 64-bit float

    std.debug.print("f32 pi: {d}\n", .{pi});
    std.debug.print("f64 e: {d}\n", .{e_value});

    // Boolean type
    const is_true: bool = true;
    const is_false: bool = false;

    std.debug.print("true: {}, false: {}\n", .{ is_true, is_false });

    // Special integer types
    const ptr_size: usize = @sizeOf(*u8); // Platform pointer size
    const index: isize = -1; // Signed pointer size

    std.debug.print("Pointer size: {d} bytes\n", .{ptr_size});
    std.debug.print("isize value: {d}\n", .{index});
}

// Type coercion and casting
pub fn type_casting() void {
    const small: u8 = 10;

    // Implicit widening (safe)
    const larger: u16 = small;
    const even_larger: u32 = larger;

    std.debug.print("Widening: {d} -> {d} -> {d}\n", .{ small, larger, even_larger });

    // Explicit casting with @intCast
    const big: u32 = 100;
    const small_again: u8 = @intCast(big);

    std.debug.print("Narrowing cast: {d} -> {d}\n", .{ big, small_again });

    // Float to int and vice versa
    const float_val: f32 = 3.7;
    const int_val: i32 = @intFromFloat(float_val);
    const back_to_float: f32 = @floatFromInt(int_val);

    std.debug.print("Float: {d}, Int: {d}, Back: {d}\n", .{ float_val, int_val, back_to_float });
}

// Arbitrary bit-width integers
pub fn arbitrary_width() void {
    const tiny: u4 = 15; // 4-bit unsigned (0-15)
    const small_signed: i5 = -16; // 5-bit signed (-16 to 15)
    const custom: u12 = 4095; // 12-bit unsigned

    std.debug.print("u4: {d}, i5: {d}, u12: {d}\n", .{ tiny, small_signed, custom });
}
