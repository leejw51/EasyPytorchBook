# やさしい Zig

*Zig プログラミング言語のシンプルで実践的なガイド*

---

## 目次

- [0. Hello World](#0-hello-world)
- [1. Zig 入門](#1-zig-入門)
- [2. 変数と型](#2-変数と型)
- [3. 制御フロー](#3-制御フロー)
- [4. 関数](#4-関数)
- [5. 配列とスライス](#5-配列とスライス)
- [6. 構造体](#6-構造体)
- [7. 列挙型と共用体](#7-列挙型と共用体)
- [8. エラー処理](#8-エラー処理)
- [9. オプショナル](#9-オプショナル)
- [10. ポインタ](#10-ポインタ)
- [11. メモリ割り当て](#11-メモリ割り当て)
- [12. コンパイル時実行](#12-コンパイル時実行)
- [13. テスト](#13-テスト)
- [14. ビルドシステム](#14-ビルドシステム)

---

## 0. Hello World


最もシンプルな Zig プログラムから始めて、次にメモリ割り当てを見ていきましょう。

## 実行方法

コードをファイル（例：`hello.zig`）に保存し、以下のコマンドで実行します：

```bash
zig run hello.zig
```

これだけです！Zig は一つのコマンドでプログラムをコンパイルして実行します。

## その他のビルド方法

```
┌─────────────────────────────────────────┐
│  zig run file.zig      # コンパイル＆実行│
│  zig build-exe file.zig # コンパイルのみ │
│  ./file                 # バイナリを実行  │
└─────────────────────────────────────────┘
```


```
Zig プログラムの実行:

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  hello.zig  │ -> │  zig run    │ -> │    出力     │
│  (ソース)   │    │ (コンパイル)│    │   (結果)    │
└─────────────┘    └─────────────┘    └─────────────┘

defer によるメモリ管理:

    ┌────────────────────────────────┐
    │  const x = try allocate();     │
    │  defer free(x);  <- 登録済み   │
    │                                │
    │  ... x を使用 ...              │
    │                                │
    └────────────────────────────────┘
              │
              ▼ (スコープ終了)
    ┌────────────────────────────────┐
    │  free(x) <- 実行される!        │
    └────────────────────────────────┘
```


### 例: hello_world

最もシンプルな Zig プログラム - コンソールに Hello World を出力

```zig
const std = @import("std");

pub fn main() void {
    std.debug.print("Hello, World!\n", .{});
}
```

**出力:**
```
Hello, World!
```


### 例: allocator_example

defer による自動クリーンアップとメモリ割り当ての総合的な例

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

**出力:**
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

## 1. Zig 入門


Zig は以下の目的で設計された現代的なシステムプログラミング言語です：
- **パフォーマンス**: 隠れた制御フローや割り当てがない
- **安全性**: オプションの安全性チェック、未定義動作がない
- **シンプルさ**: 隠れたマジックがない、読みやすいコード
- **相互運用性**: C ABI と直接互換性

## はじめに

https://ziglang.org/download/ から Zig をインストールしてください。

```
┌─────────────────────────────────────────┐
│           Zig ワークフロー               │
├─────────────────────────────────────────┤
│  1. コードを書く  →  main.zig           │
│  2. コンパイル    →  zig build-exe      │
│  3. 実行          →  ./main             │
│                                         │
│  または簡単に:    →  zig run main.zig   │
└─────────────────────────────────────────┘
```


```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   ソース     │ -> │  コンパイラ  │ -> │   バイナリ   │
│  main.zig    │    │     zig      │    │   ./main     │
└──────────────┘    └──────────────┘    └──────────────┘
```


### 例: hello_world

クラシックな Hello World プログラム

```zig
const std = @import("std");

// This is your first Zig program!
// std.debug.print outputs to stderr
pub fn main() void {
    std.debug.print("Hello, Zig!\n", .{});
}
```

**出力:**
```
Hello, Zig!
```


---

## 2. 変数と型


Zig は明示的な型を持つシンプルな型システムを持っています。

## 変数宣言

```
┌─────────────────────────────────────────┐
│  const x = 5;    // 不変 (const)        │
│  var y = 10;     // 可変 (var)          │
│  var z: i32 = 0; // 明示的な型          │
└─────────────────────────────────────────┘
```

## 基本型

| 型       | 説明              | サイズ  |
|----------|-------------------|---------|
| i8-i128  | 符号付き整数      | 1-16 B  |
| u8-u128  | 符号なし整数      | 1-16 B  |
| f32,f64  | 浮動小数点        | 4,8 B   |
| bool     | ブール値          | 1 B     |
| void     | 値なし            | 0 B     |


```
メモリレイアウト（リトルエンディアン）:

u8:   [xxxxxxxx]           (1 バイト)
u16:  [xxxxxxxx][xxxxxxxx] (2 バイト)
u32:  [xxxx][xxxx][xxxx][xxxx] (4 バイト)

i8 範囲:  -128 から 127
u8 範囲:  0 から 255
```


### 例: variables

変数宣言と基本型

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

**出力:**
```
message: Hello
count: 1
pi: 3.14
byte: 255
```


### 例: type_coercion

型変換と強制型変換

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

**出力:**
```
small: 10, big: 10
large: 1000, truncated: 232
```


---

## 3. 制御フロー


Zig はいくつかのユニークな機能を持つ標準的な制御フローを提供します。

## If/Else

```
┌─────────────────────────────────────────┐
│  if (条件) {                            │
│      // 真の場合                        │
│  } else {                               │
│      // 偽の場合                        │
│  }                                      │
└─────────────────────────────────────────┘
```

## While ループ

```
┌─────────────────────────────────────────┐
│  while (条件) : (継続式) {              │
│      // ループ本体                      │
│  }                                      │
└─────────────────────────────────────────┘
```


```
制御フロー図:

    ┌───────┐
    │ 開始  │
    └───┬───┘
        │
    ┌───▼───┐     はい   ┌───────┐
    │ 条件? ├────────────► ブロック│
    └───┬───┘            └───────┘
        │ いいえ
    ┌───▼───┐
    │  終了  │
    └───────┘
```


### 例: if_else

If/else 文と式

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

**出力:**
```
x is positive
abs(x) = 42
```


### 例: loops

While と for ループ

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

**出力:**
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


### 例: switch

Switch 式

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

**出力:**
```
Day 3: Wednesday
```


---

## 4. 関数


関数は Zig において第一級の値です。

## 関数の構文

```
┌─────────────────────────────────────────┐
│  fn 名前(引数: 型) 戻り値型 {            │
│      return 値;                         │
│  }                                      │
└─────────────────────────────────────────┘
```

## 特別な機能
- 関数を値として渡せる
- コンパイル時関数評価
- `inline` によるインライン関数


```
関数コールスタック:

┌─────────────────────┐
│   main()            │  <- スタックフレーム
├─────────────────────┤
│   add(3, 4)         │  <- 新しいフレーム
│   a = 3, b = 4      │
│   return 7          │
├─────────────────────┤
│   (main に戻る)      │
└─────────────────────┘
```


### 例: basic_functions

基本的な関数定義

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

**出力:**
```
3 + 4 = 7
17 / 5 = 3 remainder 2
```


### 例: function_pointers

関数ポインタと高階関数

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

**出力:**
```
double(5) = 10
triple(5) = 15
```


---

## 5. 配列とスライス


配列は固定サイズを持ち、スライスは配列のビューです。

## 配列 vs スライス

```
┌─────────────────────────────────────────┐
│  配列: [N]T    - コンパイル時固定サイズ │
│  スライス: []T - 実行時配列ビュー       │
└─────────────────────────────────────────┘
```


```
メモリ内の配列:
┌───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │  arr[5]i32
└───┴───┴───┴───┴───┘
  0   1   2   3   4

スライス（ビュー）:
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


### 例: arrays

配列の基本

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

**出力:**
```
arr1[0] = 1
arr2.len = 3
sum of arr1 = 15
```


### 例: slices

スライス - 配列のビュー

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

**出力:**
```
slice.len = 3
slice[0] = 2
slice[1] = 3
slice[2] = 4
```


### 例: string_literals

バイトスライスとしての文字列

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

**出力:**
```
String: Hello, World!
Length: 13
First char: H
H e l l o
```


---

## 6. 構造体


構造体は関連するデータをまとめます。

## 構造体の定義

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
構造体のメモリレイアウト:

Point { x: i32, y: i32 }

┌─────────────┬─────────────┐
│    x: i32   │    y: i32   │
│   4 バイト  │   4 バイト  │
└─────────────┴─────────────┘
     合計: 8 バイト
```


### 例: basic_struct

メソッドを持つ構造体の定義

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

**出力:**
```
Point: (3, 4)
Distance: 5.00
```


### 例: struct_defaults

デフォルト値を持つ構造体

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

**出力:**
```
c1: 800x600 'Untitled'
c2: 1920x1080 'Untitled'
```


---

## 7. 列挙型と共用体


列挙型は名前付きの値のセットを定義します。タグ付き共用体は列挙型とデータを組み合わせます。

## 列挙型

```
┌─────────────────────────────────────────┐
│  const Color = enum { red, green, blue };│
└─────────────────────────────────────────┘
```

## タグ付き共用体

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
タグ付き共用体のメモリ:

Value union(enum):
┌─────────┬─────────────────┐
│  タグ   │     データ      │
│ (enum)  │ (最大フィールド)│
└─────────┴─────────────────┘

タグ値: .int=0, .float=1, .boolean=2, .none=3
```


### 例: enums

メソッドを持つ列挙型

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

**出力:**
```
Direction: .north
Opposite: .south
```


### 例: tagged_union

型安全なバリアントのためのタグ付き共用体

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

**出力:**
```
int: 42
float: 3.14
bool: true
none
```


---

## 8. エラー処理


Zig はエラーユニオンを使用した明示的なエラー処理を使用します。

## エラーユニオン型

```
┌─────────────────────────────────────────┐
│  fn divide(a: i32, b: i32) !i32 {       │
│      if (b == 0) return error.DivByZero;│
│      return @divTrunc(a, b);            │
│  }                                      │
└─────────────────────────────────────────┘
```

## エラー処理オプション
- `try`: エラーを上位に伝播
- `catch`: エラーを処理
- `orelse`: エラー時のデフォルト値


```
エラーユニオン型:

    anyerror!T または ErrorSet!T
         │
    ┌────┴────┐
    │         │
  エラー      値
    │         │
┌───▼───┐ ┌───▼───┐
│ catch │ │ 成功  │
└───────┘ └───────┘
```


### 例: error_basics

catch を使った基本的なエラー処理

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

**出力:**
```
10 / 2 = 5
10 / 0 = 0 (default)
```


### 例: try_keyword

try を使ったエラー伝播

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

**出力:**
```
Parsed: 123
Parse error: error.InvalidChar
```


---

## 9. オプショナル


オプショナルは存在するかもしれないし、しないかもしれない値を表します。

## オプショナル型

```
┌─────────────────────────────────────────┐
│  var x: ?i32 = null;   // 値なし        │
│  x = 42;               // 値あり        │
│                                         │
│  if (x) |val| {        // アンラップ    │
│      // val を使用                      │
│  }                                      │
└─────────────────────────────────────────┘
```


```
オプショナル型 ?T:

┌─────────────┐
│   ?i32      │
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
 null     値
   │       │
┌──▼──┐ ┌──▼──┐
│ 空  │ │ 42  │
└─────┘ └─────┘
```


### 例: optionals

オプショナル型とアンラップ

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

**出力:**
```
First even: 4
Result with default: -1
```


### 例: optional_pointers

リンク構造のためのオプショナルポインタ

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

**出力:**
```
Value: 10
Value: 20
Value: 30
```


---

## 10. ポインタ


Zig は異なる用途のための複数のポインタ型を持っています。

## ポインタ型

```
┌─────────────────────────────────────────┐
│  *T       - 単一アイテムポインタ        │
│  [*]T     - 複数アイテムポインタ        │
│  *const T - const へのポインタ          │
│  ?*T      - オプショナルポインタ        │
└─────────────────────────────────────────┘
```


```
変数へのポインタ:

   ptr          x
┌───────┐   ┌───────┐
│アドレス┼──►│  10   │
└───────┘   └───────┘

*ptr は値を取得するためにデリファレンス
&x は x のアドレスを取得
```


### 例: pointers_basic

基本的なポインタ操作

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

**出力:**
```
x = 10
*ptr = 10
After modification: x = 20
```


### 例: pointer_arithmetic

ポインタ演算とスライス

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

**出力:**
```
ptr[0] = 10
ptr[2] = 30
slice val: 20
slice val: 30
slice val: 40
```


---

## 11. メモリ割り当て


Zig はメモリ割り当てに対する明示的な制御を提供します。

## アロケータインターフェース

```
┌─────────────────────────────────────────┐
│  const allocator = std.heap.page_allocator;│
│  const ptr = try allocator.create(T);   │
│  defer allocator.destroy(ptr);          │
└─────────────────────────────────────────┘
```

## 一般的なアロケータ
- `page_allocator`: OS ページ割り当て
- `GeneralPurposeAllocator`: デバッグフレンドリー
- `ArenaAllocator`: 一括解放


```
メモリ管理:

スタック（自動）:
┌─────────────────┐
│  ローカル変数   │ <- 高速、自動クリーンアップ
└─────────────────┘

ヒープ（手動）:
┌─────────────────┐
│ allocator.alloc │ <- 明示的割り当て
│ allocator.free  │ <- 明示的解放
└─────────────────┘

クリーンアップには defer を使用！
```


### 例: allocation

動的メモリ割り当て

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

**出力:**
```
0 10 20 30 40
```


### 例: arraylist

動的配列のための ArrayList の使用

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

**出力:**
```
List: 10 20 30 
Length: 3
```


---

## 12. コンパイル時実行


Zig は `comptime` を使用してコンパイル時にコードを実行できます。

## Comptime の機能

```
┌─────────────────────────────────────────┐
│  comptime {                             │
│      // コンパイル時に実行              │
│  }                                      │
│                                         │
│  fn generic(comptime T: type) type {    │
│      // 型レベルの計算                  │
│  }                                      │
└─────────────────────────────────────────┘
```


```
コンパイル時 vs 実行時:

┌──────────────────┐     ┌──────────────────┐
│   コンパイル時   │     │     実行時       │
├──────────────────┤     ├──────────────────┤
│ comptime ブロック│     │ 通常のコード     │
│ 型の計算         │     │ ユーザー入力     │
│ ジェネリック引数 │     │ 動的データ       │
│ 定数畳み込み     │     │ ヒープ割り当て   │
└──────────────────┘     └──────────────────┘
        │                        │
        └────────┬───────────────┘
                 │
            最終バイナリ
```


### 例: comptime_basic

コンパイル時計算

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

**出力:**
```
10! = 3628800
Size of i32: 4 bytes
Squares: 0 1 4 9 16
```


### 例: generic_function

comptime を使ったジェネリック関数と型

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

**出力:**
```
max(3, 7) = 7
max(3.5, 2.1) = 3.5
Before swap: 10, 20
After swap: 20, 10
```


---

## 13. テスト


Zig には組み込みのテストサポートがあります。

## テスト構文

```
┌─────────────────────────────────────────┐
│  test "説明" {                          │
│      try std.testing.expect(true);      │
│      try std.testing.expectEqual(1, 1); │
│  }                                      │
└─────────────────────────────────────────┘
```

テストを実行: `zig test file.zig`


```
テストワークフロー:

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ テスト作成  │ -> │  zig test   │ -> │    結果     │
└─────────────┘    └─────────────┘    └─────────────┘
                          │
                   ┌──────┴──────┐
                   │             │
                 成功          失敗
                   │             │
                  [OK]     [エラーメッセージ]
```


### 例: testing

テストの作成と実行

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

**出力:**
```
Run 'zig test' to execute tests
add(2,3) = 5
```


---

## 14. ビルドシステム


Zig には `build.zig` を使用する強力な組み込みビルドシステムがあります。

## 基本的な build.zig

```
┌─────────────────────────────────────────┐
│  zig init     - 新しいプロジェクトを作成│
│  zig build    - プロジェクトをビルド    │
│  zig build run - ビルドして実行         │
└─────────────────────────────────────────┘
```


```
プロジェクト構造:

my_project/
├── build.zig       <- ビルド設定
├── build.zig.zon   <- 依存関係
└── src/
    ├── main.zig    <- エントリーポイント
    └── lib.zig     <- ライブラリコード

ビルドプロセス:
┌────────────┐   ┌────────────┐   ┌────────────┐
│ build.zig  │ → │ zig build  │ → │ zig-out/   │
└────────────┘   └────────────┘   └────────────┘
```


### 例: build_example

プロジェクト構造とビルドコマンド

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

**出力:**
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

## クイックリファレンス


### 一般的なコマンド
```
zig run file.zig     # コンパイルして実行
zig build-exe file.zig  # 実行ファイルにコンパイル
zig test file.zig    # テストを実行
zig fmt file.zig     # コードをフォーマット
zig init             # 新しいプロジェクトを初期化
zig build            # プロジェクトをビルド
```

### 便利な組み込み関数
| 組み込み関数 | 説明 |
|----------|-------------|
| @import | モジュールをインポート |
| @intCast | 整数型キャスト |
| @floatFromInt | 整数から浮動小数点に変換 |
| @truncate | より小さい型に切り詰め |
| @sizeOf | 型のサイズを取得 |
| @TypeOf | 式の型を取得 |

### フォーマット指定子
| 指定子 | 説明 |
|-----------|-------------|
| {} | デフォルト |
| {s} | 文字列 |
| {c} | 文字 |
| {d} | 10進数 |
| {x} | 16進数 |
| {b} | 2進数 |
