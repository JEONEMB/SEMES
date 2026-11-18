ENTRY(TestTransport)
MEMORY {
  FLASH (rx) : ORIGIN = 0x00010000, LENGTH = 1M
  RAM (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}
SECTIONS {
  .text : { *(.text*) *(.rodata*) } > FLASH
  .data : { *(.data*) } > RAM
  .bss (NOLOAD) : { *(.bss*) *(COMMON) } > RAM
}
