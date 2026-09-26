package com.productdiscovery.service;

import com.productdiscovery.domain.Product;
import com.productdiscovery.repository.ProductRepository;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;
import java.io.Reader;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.List;

@Component
public class CatalogImportRunner implements ApplicationRunner {
    private final ProductRepository products;
    private final String csvPath;
    private final String imagesDir;

    public CatalogImportRunner(ProductRepository products,
            @Value("${catalog.csv-path:}") String csvPath,
            @Value("${catalog.images-dir:images}") String imagesDir) {
        this.products = products;
        this.csvPath = csvPath;
        this.imagesDir = imagesDir;
    }

    @Override
    @Transactional
    public void run(ApplicationArguments args) throws Exception {
        if (csvPath == null || csvPath.isBlank() || products.count() > 0) return;
        Path path = Path.of(csvPath);
        if (!Files.isRegularFile(path)) {
            throw new IllegalStateException("CATALOG_CSV_PATH does not exist: " + path.toAbsolutePath());
        }
        try (Reader reader = Files.newBufferedReader(path)) {
            Iterable<CSVRecord> records = CSVFormat.DEFAULT.builder()
                    .setHeader().setSkipHeaderRecord(true).build().parse(reader);
            List<Product> batch = new ArrayList<>(1000);
            long mlIndex = 0;
            for (CSVRecord row : records) {
                long productId = Long.parseLong(row.get("id"));
                batch.add(new Product(productId, mlIndex++, value(row, "productDisplayName"),
                        value(row, "articleType"), value(row, "baseColour"),
                        value(row, "season"), value(row, "usage"), value(row, "gender"),
                        Path.of(imagesDir, productId + ".jpg").toString()));
                if (batch.size() == 1000) { products.saveAll(batch); batch.clear(); }
            }
            if (!batch.isEmpty()) products.saveAll(batch);
        }
    }

    private static String value(CSVRecord row, String key) {
        return row.isMapped(key) && row.isSet(key) ? row.get(key) : null;
    }
}
